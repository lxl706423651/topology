#!/usr/bin/env python3
"""Export BIRD routes to Linux FIB and verify one fixed router in every AS.

Inputs: config.yaml, Compose and the successful startBird.json/PID cache in the
same experiment directory. Outputs: config backups and startKernel.json.
Requires root; reconfigures router containers only, then waits for load1 < 50.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from runExperiment import ExperimentContext, ROUTER_RE, addStageArguments, writeJson


def enableKernelExport(text: str) -> str:
    """Change text's sole export clause to export all, preserving template parameters."""
    pattern = re.compile(r'#[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\b[A-Za-z_]\w*\b|[{};]', re.DOTALL)
    tokens = [match for match in pattern.finditer(text)
              if not match.group().startswith(("#", "/*", '"'))]
    exports = [index for index, token in enumerate(tokens) if token.group() == "export"]
    if len(exports) != 1:
        raise ValueError("kernel.conf must contain one IPv4 export clause")
    index = exports[0]
    depth = 0
    for token in tokens[index + 1:]:
        value = token.group()
        if value == "{":
            depth += 1
        elif value == "}":
            depth -= 1
            if depth < 0:
                break
        elif value == ";" and depth == 0:
            return text[:tokens[index].start()] + "export all;" + text[token.end():]
    raise ValueError("unterminated export clause in kernel.conf")


def loadVerifiedContainers(context: ExperimentContext) -> dict:
    """Require context's successful BIRD gate and reject stale container IDs/PIDs."""
    cache = json.loads((context.runtime / "containerPids.json").read_text())
    report = json.loads((context.directory / "startBird.json").read_text())
    for record in (cache, report):
        if record.get("project") != context.project or record.get("compose_sha256") != context.fingerprint:
            raise ValueError("BIRD cache/report belongs to another project or Compose file; rerun startBird.py")
    if not report.get("verify_after_start_bird", {}).get("success"):
        raise ValueError("the BIRD verification gate has not passed; rerun startBird.py")
    current = context.discoverContainers()
    if cache.get("containers") != current:
        raise ValueError("container IDs or PIDs changed; rerun startBird.py in this experiment directory")
    return {name: node for name, node in current.items() if name in context.inventory["routers"]}


def updateRouter(context: ExperimentContext, name: str, pid: str) -> None:
    """Back up name's kernel configuration, export routes in pid, and restore on failure."""
    previous = context.runInNamespace(pid, "cat", "/etc/bird/conf/kernel.conf")
    previous.check_returncode()
    backups = context.runtime / "kernelConfigs"
    backups.mkdir(exist_ok=True)
    backup = backups / f"{name}.conf"
    if not backup.exists():
        backup.write_text(previous.stdout)
    # The initial file comes from RoutingFileTemplates["rnode_kernel_conf1"].
    # Preserve the user's scan interval, protocol name and other experiment knobs.
    text = enableKernelExport(previous.stdout)
    try:
        context.runInNamespace(pid, "tee", "/etc/bird/conf/kernel.conf", inputText=text).check_returncode()
        context.runInNamespace(pid, "bird", "-p", "-c", "/etc/bird/bird.conf").check_returncode()
        configured = context.runInNamespace(pid, "birdc", "configure")
        if configured.returncode:
            raise RuntimeError(f"{name}: {configured.stdout}{configured.stderr}")
        # Reconfigure applies the changed export filter. No invalid `reload kernel`
        # command is needed, and the template's protocol name is preserved.
    except Exception:
        context.runInNamespace(pid, "tee", "/etc/bird/conf/kernel.conf", inputText=previous.stdout)
        context.runInNamespace(pid, "birdc", "configure")
        raise


def startKernelExport(context: ExperimentContext) -> dict:
    """Enable FIB exports in context's verified router set and record the separate FIB gate."""
    result = {"compose_sha256": context.fingerprint, "project": context.project,
              "start_kernel": {"success": False}, "verify_after_fib_write": {"success": False}}
    started = time.monotonic()
    try:
        if os.geteuid() != 0:
            raise PermissionError("startKernel.py requires root")
        containers = loadVerifiedContainers(context)
        names = sorted(containers, key=lambda name: (*map(int, ROUTER_RE.match(name).groups()), name))
        for index, name in enumerate(names, 1):
            if os.getloadavg()[0] > context.settings["kernel"]["load_threshold"]:
                context.waitForLoad(context.settings["kernel"]["recovery_threshold"])
            updateRouter(context, name, containers[name]["pid"])
            result["start_kernel"]["processed"] = index
            if index % 100 == 0:
                context.writeLog(f"Kernel export: {index}/{len(names)} routers processed")
            time.sleep(context.settings["kernel"]["update_interval"])
        result["start_kernel"].update(success=True, duration_seconds=round(time.monotonic() - started, 3))
        result["verify_after_fib_write"] = context.verifyState(fib=True)
        if not result["verify_after_fib_write"]["success"]:
            raise RuntimeError("FIB verification failed; see startKernel.json for per-AS counts and raw outputs")
        context.writeLog(f"Kernel export completed on {len(names)} routers; all AS representatives passed")
    except Exception as error:
        result["error"] = str(error)
        context.writeLog(f"ERROR: {error}")
    finally:
        result["success"] = result["start_kernel"]["success"] and result["verify_after_fib_write"]["success"]
        writeJson(context.directory / "startKernel.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    addStageArguments(parser)
    args = parser.parse_args()
    raise SystemExit(0 if startKernelExport(ExperimentContext(args))["success"] else 1)
