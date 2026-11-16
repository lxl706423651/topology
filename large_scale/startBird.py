#!/usr/bin/env python3
"""Start BIRD and verify fixed control-plane samples in the selected Compose project.

Inputs: config.yaml and output/docker-compose.yml. Outputs: PID cache and
startBird.json in the experiment directory. Requires root for nsenter; starts
BIRD inside existing containers, then waits for load1 < 50 before verification.
"""

from __future__ import annotations

import argparse
import os
import time

from runExperiment import ExperimentContext, addStageArguments, writeJson


def startBirds(context: ExperimentContext) -> dict:
    """Cache context's exact container set, start missing daemons, and run its BIRD gate."""
    result = {"compose_sha256": context.fingerprint, "project": context.project,
              "start_bird": {"success": False}, "verify_after_start_bird": {"success": False}}
    started = time.monotonic()
    try:
        if os.geteuid() != 0:
            raise PermissionError("startBird.py requires root: use sudo with the activated Python interpreter")
        containers = context.discoverContainers()
        writeJson(context.runtime / "containerPids.json", {"project": context.project,
                  "compose_sha256": context.fingerprint, "containers": containers})
        failures = []
        for index, (name, node) in enumerate(sorted(containers.items()), 1):
            pid = node["pid"]
            if context.runInNamespace(pid, "birdc", "show", "status").returncode == 0:
                continue
            parsed = context.runInNamespace(pid, "bird", "-p", "-c", "/etc/bird/bird.conf")
            launch = context.runInNamespace(pid, "bird") if parsed.returncode == 0 else parsed
            if launch.returncode:
                failures.append({"name": name, "output": launch.stdout + launch.stderr})
            if index % 200 == 0:
                context.writeLog(f"BIRD startup: {index}/{len(containers)} processed")
            time.sleep(context.settings["bird"]["start_delay"])
        result["start_bird"].update(processed=len(containers), failed_samples=failures)
        if failures:
            raise RuntimeError(f"BIRD startup failed in {len(failures)} containers")
        time.sleep(context.settings["bird"]["post_start_delay"])
        result["start_bird"].update(success=True, duration_seconds=round(time.monotonic() - started, 3))
        result["verify_after_start_bird"] = context.verifyState(fib=False)
        if not result["verify_after_start_bird"]["success"]:
            raise RuntimeError("BIRD protocol verification failed; see startBird.json for raw failing samples")
        context.writeLog(f"BIRD started in {len(containers)} nodes; fixed AS samples passed")
    except Exception as error:
        result["error"] = str(error)
        context.writeLog(f"ERROR: {error}")
    finally:
        result["success"] = result["start_bird"]["success"] and result["verify_after_start_bird"]["success"]
        writeJson(context.directory / "startBird.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    addStageArguments(parser)
    args = parser.parse_args()
    raise SystemExit(0 if startBirds(ExperimentContext(args))["success"] else 1)
