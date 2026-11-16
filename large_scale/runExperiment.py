#!/usr/bin/env python3
"""Run compile -> image build -> Docker up -> BIRD -> kernel export.

Inputs: config.yaml, topology/input and the selected compiled Compose file.
Outputs: output plus per-stage logs and summaries under evaluationResult.
Full execution builds images and starts containers locally. The up subcommand
is used by startDocker.sh. Shared stage helpers live here; importing this file
only defines configuration/runtime helpers. Use the seedpy310 interpreter.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULTS = {
    "topology": {"scale": 3083, "input": "topology/input", "output": "output", "platform": "amd"},
    "docker": {"project": "large_scale", "parallel_jobs": 8, "batch_size": 50, "ready_timeout": 900},
    "bird": {"start_delay": 0.02, "post_start_delay": 60},
    "kernel": {"load_threshold": 110, "recovery_threshold": 50, "update_interval": 0.5},
    "verification": {"load_threshold": 50, "check_interval": 30, "command_timeout": 60},
}
ROUTER_RE = re.compile(r"^as(\d+)brd-r(\d+)(?:-|$)")
ROLE_LABEL = "org.seedsecuritylabs.seedemu.meta.role"


def loadSettings(configPath: Path | None = None) -> dict:
    """Read configPath, reject unknown fields, and resolve paths beside that YAML."""
    path = (configPath or SCRIPT_DIR / "config.yaml").expanduser().resolve()
    supplied = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = copy.deepcopy(DEFAULTS)
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("config.yaml contains an unknown section")
    for section, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(DEFAULTS[section]):
            raise ValueError(f"unknown fields in configuration section {section}")
        settings[section].update(values)
    positive = [("topology", "scale"), ("docker", "parallel_jobs"), ("docker", "batch_size"),
                ("docker", "ready_timeout"), ("verification", "check_interval"),
                ("verification", "command_timeout"), ("verification", "load_threshold"),
                ("kernel", "load_threshold"), ("kernel", "recovery_threshold")]
    for section, key in positive:
        value = settings[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    for section, key in [("topology", "scale"), ("docker", "parallel_jobs"), ("docker", "batch_size")]:
        if not isinstance(settings[section][key], int):
            raise ValueError(f"{section}.{key} must be an integer")
    for section, key in [("bird", "start_delay"), ("bird", "post_start_delay"), ("kernel", "update_interval")]:
        value = settings[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{section}.{key} must be nonnegative")
    if settings["verification"]["load_threshold"] > 50:
        raise ValueError("verification.load_threshold must be <= 50")
    if settings["kernel"]["recovery_threshold"] > settings["kernel"]["load_threshold"]:
        raise ValueError("kernel.recovery_threshold must not exceed load_threshold")
    if settings["topology"]["platform"] not in {"amd", "arm"}:
        raise ValueError("topology.platform must be amd or arm")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(settings["docker"]["project"])):
        raise ValueError("docker.project must be a valid Compose project name")
    for key in ("input", "output"):
        value = Path(settings["topology"][key]).expanduser()
        settings["topology"][key] = str((path.parent / value).resolve())
    settings["config_path"] = str(path)
    return settings


def writeJson(path: Path, value: dict) -> None:
    """Atomically write value to path; create its parent runtime directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def readCompose(composePath: Path) -> dict:
    """Read composePath and derive BIRD/router counts from generated metadata."""
    data = yaml.safe_load(composePath.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("services"), dict):
        raise ValueError(f"invalid Compose file: {composePath}")
    nodes, routers = {}, {}
    for service, spec in data["services"].items():
        name = str(spec.get("container_name", ""))
        role = spec.get("labels", {}).get(ROLE_LABEL)
        if role in {"Router", "BorderRouter", "Route Server"} or name.startswith("as"):
            nodes[name] = service
            if ROUTER_RE.match(name):
                routers[name] = service
    if not nodes or not routers:
        raise ValueError("Compose file contains no named BIRD routers")
    return {"data": data, "nodes": nodes, "routers": routers,
            "counts": {"services": len(data["services"]), "bird_nodes": len(nodes),
                       "routers": len(routers), "route_servers": len(nodes) - len(routers),
                       "networks": len(data.get("networks", {}))}}


def selectRepresentatives(names: list[str], limit: int | None = None) -> list[str]:
    """Select the smallest numeric router number per ASN from names, then limit ASNs."""
    groups = {}
    for name in names:
        match = ROUTER_RE.match(name)
        if match:
            asn, number = map(int, match.groups())
            groups[asn] = min(groups.get(asn, (number, name)), (number, name))
    asns = sorted(groups)
    return [groups[asn][1] for asn in (asns if limit is None else asns[:limit])]


def parseNetworkCount(text: str) -> tuple[int, str]:
    """Parse BIRD text, preferring selected-network counts over alternative routes."""
    match = re.search(r"\bfor\s+(\d+)\s+networks\b", text)
    if match:
        return int(match.group(1)), "networks"
    match = re.search(r"(?m)^\s*(\d+)\s+routes\b", text)
    if match:
        return int(match.group(1)), "routes_fallback"
    raise ValueError("cannot parse BIRD route count")


def findBadProtocols(text: str) -> list[str]:
    """Return unhealthy protocol rows in BIRD text; BGP must be Established."""
    rows = []
    bad = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] not in {"BGP", "OSPF", "Device", "Direct", "Kernel", "Pipe", "Static"}:
            continue
        rows.append(line)
        if parts[3].lower() != "up" or (parts[1] == "BGP" and "Established" not in parts[4:]):
            bad.append(line)
    return bad if rows else ["No protocol rows returned by BIRD"]


class ExperimentContext:
    """Bind settings, a Compose file and an experiment directory to one project."""

    def __init__(self, args: argparse.Namespace):
        self.settings = loadSettings(args.config)
        self.compose = (args.compose or Path(self.settings["topology"]["output"]) / "docker-compose.yml").resolve()
        self.project = args.project_name or self.settings["docker"]["project"]
        self.directory = (args.experiment_dir or SCRIPT_DIR / "evaluationResult" /
                          datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runtime = self.directory / "runtime"
        self.runtime.mkdir(exist_ok=True)
        self.inventory = readCompose(self.compose)
        self.fingerprint = hashlib.sha256(self.compose.read_bytes()).hexdigest()

    def writeLog(self, message: str) -> None:
        """Print message and append it to this experiment's common log."""
        line = f"[{datetime.datetime.now():%F %T}] {message}"
        print(line, flush=True)
        with (self.directory / "runExperiment.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def runCommand(self, command: list[str], *, inputText: str | None = None) -> subprocess.CompletedProcess:
        """Run command with bounded time and optional stdin inputText, capturing evidence."""
        return subprocess.run(command, input=inputText, capture_output=True, text=True,
                              timeout=self.settings["verification"]["command_timeout"], check=False)

    def runInNamespace(self, pid: str, *command: str, inputText: str | None = None) -> subprocess.CompletedProcess:
        """Run command in pid's network/mount namespaces, optionally supplying inputText."""
        return self.runCommand(["nsenter", "-t", str(pid), "-n", "-m", *command], inputText=inputText)

    def discoverContainers(self) -> dict:
        """Inspect only running containers in this project and require the exact node set."""
        result = self.runCommand(["docker", "ps", "-q", "--filter", f"label=com.docker.compose.project={self.project}"])
        result.check_returncode()
        ids = result.stdout.split()
        found = {}
        for offset in range(0, len(ids), 200):
            inspected = self.runCommand(["docker", "inspect", *ids[offset:offset + 200]])
            inspected.check_returncode()
            for row in json.loads(inspected.stdout):
                name = row["Name"].lstrip("/")
                if name in self.inventory["nodes"] and row["State"]["Running"]:
                    found[name] = {"pid": str(row["State"]["Pid"]), "id": row["Id"]}
        missing = set(self.inventory["nodes"]) - set(found)
        if missing:
            raise RuntimeError(f"{len(missing)} BIRD containers are not running: {sorted(missing)[:10]}")
        return found

    def waitForLoad(self, threshold: float | None = None) -> dict:
        """Wait for local one-minute load below threshold (default verification gate)."""
        threshold = threshold or self.settings["verification"]["load_threshold"]
        started = time.monotonic()
        while True:
            load = os.getloadavg()[0]
            if load < threshold:
                return {"load1": load, "load_wait_seconds": round(time.monotonic() - started, 3)}
            self.writeLog(f"Waiting for load1 < {threshold}: current={load:.2f}")
            time.sleep(self.settings["verification"]["check_interval"])

    def verifyState(self, fib: bool) -> dict:
        """After a load gate, verify fixed per-AS samples; fib selects the broader FIB gate."""
        started = time.monotonic()
        gate = self.waitForLoad()
        names = selectRepresentatives(list(self.inventory["routers"]), None if fib else 10)
        rows = []
        for name in names:
            sample = {"name": name, "asn": int(ROUTER_RE.match(name).group(1)), "success": False}
            try:
                route = self.runCommand(["docker", "exec", name, "birdc", "show", "route", "count"])
                sample["route_output"] = route.stdout + route.stderr
                route.check_returncode()
                count, mode = parseNetworkCount(route.stdout)
                sample.update(bird_network_count=count, parse_mode=mode)
                if fib:
                    other = self.runCommand(["docker", "exec", name, "sh", "-lc", "ip route | wc -l"])
                    sample["fib_output"] = other.stdout + other.stderr
                    other.check_returncode()
                    sample["ip_route_wc_count"] = int(other.stdout.strip())
                    sample["success"] = count < sample["ip_route_wc_count"]
                else:
                    other = self.runCommand(["docker", "exec", name, "birdc", "show", "protocols"])
                    sample["protocol_output"] = other.stdout + other.stderr
                    other.check_returncode()
                    sample["failed_protocol_lines"] = findBadProtocols(other.stdout)
                    sample["success"] = not sample["failed_protocol_lines"]
            except (ValueError, subprocess.SubprocessError, OSError) as error:
                sample["error"] = str(error)
            rows.append(sample)
        result = {"success": bool(rows) and all(row["success"] for row in rows), **gate,
                  "duration_seconds": round(time.monotonic() - started, 3), "samples": rows,
                  "failed_samples": [row for row in rows if not row["success"]]}
        result["as_count" if fib else "sample_count"] = len(rows)
        result["checked_containers" if fib else "selected_containers"] = names
        return result


def addStageArguments(parser: argparse.ArgumentParser) -> None:
    """Add common config, Compose, project and log-directory options to parser."""
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--compose", type=Path)
    parser.add_argument("--project-name")
    parser.add_argument("--experiment-dir", type=Path)


def startDocker(context: ExperimentContext) -> None:
    """Start context's services in bounded parallel Compose batches and wait for nodes."""
    config = context.settings["docker"]
    command = ["docker", "compose", "-p", context.project, "-f", str(context.compose)]
    subprocess.run(command + ["config", "--quiet"], check=True)
    services = list(context.inventory["data"]["services"])
    batches = [services[i:i + config["batch_size"]] for i in range(0, len(services), config["batch_size"])]
    logDirectory = context.directory / "dockerUp"
    logDirectory.mkdir(exist_ok=True)

    def startBatch(index: int, batch: list[str]) -> int:
        """Start service batch and write its output to a unique index log."""
        with (logDirectory / f"batch_{index:04d}.log").open("w") as log:
            return subprocess.run(command + ["up", "-d", "--no-build", *batch], stdout=log,
                                  stderr=subprocess.STDOUT, check=False).returncode

    failures = []
    with ThreadPoolExecutor(max_workers=config["parallel_jobs"]) as pool:
        pending = {pool.submit(startBatch, i, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(pending):
            if future.result() != 0:
                failures.append(pending[future])
    if failures:
        raise RuntimeError(f"Compose startup failed in batches {sorted(failures)}; see {logDirectory}")
    deadline = time.monotonic() + config["ready_timeout"]
    while True:
        try:
            nodes = context.discoverContainers()
            context.writeLog(f"All {len(nodes)} BIRD containers are running")
            return
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"container readiness timed out: {error}") from error
            context.writeLog(str(error))
            time.sleep(5)


def executeStage(command: list[str], logPath: Path) -> None:
    """Run command with the current interpreter on PATH and tee output to logPath."""
    environment = dict(os.environ)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    logPath.parent.mkdir(parents=True, exist_ok=True)
    with logPath.open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + repr(command) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=environment) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def checkHost(settings: dict, scale: int) -> dict:
    """Validate settings' chosen input scale and the local Docker/namespace tools."""
    from large_internrt import createTopologyPlan

    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 is required; activate seedpy310")
    plan = createTopologyPlan(Path(settings["topology"]["input"]), scale)
    for command in ("docker", "nsenter"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required command is missing: {command}")
    for command in (["docker", "info"], ["docker", "compose", "version"]):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=60)
    return plan["counts"]


def runSelectedExperiment() -> int:
    """Parse the selected scale/resume stage and record each stage's outcome."""
    stages = ["compile", "build", "up", "bird", "kernel"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--scale", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--from-stage", choices=stages, default="compile")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace only a recognized previous compiler output")
    args = parser.parse_args()
    settings = loadSettings(args.config)
    output = (args.output or Path(settings["topology"]["output"])).resolve()
    compose = output / "docker-compose.yml"
    scale = args.scale if args.scale is not None else settings["topology"]["scale"]
    metadataPath = output / "topology-summary.json"
    if args.from_stage != "compile":
        metadata = json.loads(metadataPath.read_text())
        compiledScale = metadata["counts"]["scale"]
        if args.scale is not None and scale != compiledScale:
            raise ValueError(f"requested scale {scale} differs from compiled output {compiledScale}")
        scale = compiledScale
    if args.from_stage == "kernel" and args.experiment_dir is None:
        parser.error("--from-stage kernel requires the original --experiment-dir")
    directory = (args.experiment_dir or SCRIPT_DIR / "evaluationResult" /
                 f"{scale}nodes_{datetime.datetime.now():%Y%m%d_%H%M%S}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Create the shared log as the invoking user before root namespace stages.
    (directory / "runExperiment.log").touch(exist_ok=True)
    summaryPath = directory / "summary.json"
    previous = json.loads(summaryPath.read_text()) if summaryPath.exists() else {}
    if previous and (previous.get("scale") != scale or previous.get("compose") != str(compose)):
        raise ValueError("experiment directory belongs to a different scale or output")
    summary = {"scale": scale, "compose": str(compose), "project": settings["docker"]["project"],
               "python": sys.executable, "config": settings["config_path"], "success": False,
               "started_at": datetime.datetime.now().astimezone().isoformat(),
               "stages": previous.get("stages", {})}
    started = time.monotonic()
    currentStage = "preflight"
    try:
        summary["input_counts"] = checkHost(settings, scale)
        if args.preflight_only:
            if compose.is_file():
                subprocess.run(["docker", "compose", "-p", settings["docker"]["project"],
                                "-f", str(compose), "config", "--quiet"], check=True)
            summary.update(success=True, preflight_only=True)
            print(f"Preflight passed for scale {scale}. Logs: {directory}")
            return 0
        print(f"Scale: {scale}; logs: {directory}", flush=True)
        common = ["--config", settings["config_path"], "--compose", str(compose),
                  "--project-name", settings["docker"]["project"], "--experiment-dir", str(directory)]
        for stage in stages[stages.index(args.from_stage):]:
            currentStage = stage
            stageStarted = time.monotonic()
            summary["stages"][stage] = {"success": False}
            writeJson(summaryPath, summary)
            if stage == "compile":
                command = [sys.executable, str(SCRIPT_DIR / "large_internrt.py"), str(scale),
                           "--config", settings["config_path"], "--output", str(output)]
                if args.overwrite:
                    command.append("--overwrite")
            elif stage == "build":
                command = [sys.executable, str(SCRIPT_DIR / "buildDockerImages.py"), str(compose),
                           "--project-name", settings["docker"]["project"], "--log-dir", str(directory / "dockerBuild")]
            elif stage == "up":
                command = [str(SCRIPT_DIR / "startDocker.sh"), *common]
            else:
                filename = "startBird.py" if stage == "bird" else "startKernel.py"
                command = [sys.executable, str(SCRIPT_DIR / filename), *common]
                if os.geteuid() != 0:
                    if shutil.which("sudo") is None:
                        raise RuntimeError("sudo is required for namespace stages")
                    command = ["sudo", "--", *command]
            try:
                executeStage(command, directory / f"{stage}.log")
            finally:
                if stage in {"bird", "kernel"}:
                    reportPath = directory / ("startBird.json" if stage == "bird" else "startKernel.json")
                    if reportPath.exists():
                        report = json.loads(reportPath.read_text())
                        for key in ("start_bird", "verify_after_start_bird", "start_kernel", "verify_after_fib_write"):
                            if key in report:
                                summary["stages"][key] = report[key]
            summary["stages"][stage].update(success=True, duration_seconds=round(time.monotonic() - stageStarted, 3))
            if stage == "compile":
                inventory = readCompose(compose)
                if inventory["counts"]["bird_nodes"] != scale:
                    raise ValueError("compiled Compose node count differs from the requested scale")
                summary["compose_counts"] = inventory["counts"]
            writeJson(summaryPath, summary)
        summary["success"] = True
        print(f"Experiment and both verification gates passed. Logs: {directory}")
        return 0
    except Exception as error:
        summary.update(error=str(error), failed_stage=currentStage)
        print(f"ERROR in {currentStage}: {error}\nLogs: {directory}", file=sys.stderr)
        return 1
    finally:
        summary["duration_seconds"] = round(time.monotonic() - started, 3)
        writeJson(summaryPath, summary)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "up":
        parser = argparse.ArgumentParser(prog="startDocker.sh", description="Start Compose services in batches")
        addStageArguments(parser)
        arguments = parser.parse_args(sys.argv[2:])
        startDocker(ExperimentContext(arguments))
    else:
        raise SystemExit(runSelectedExperiment())
