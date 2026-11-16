#!/usr/bin/env python3
"""Compare docker exec and nsenter command-entry overhead on SeedEMU containers."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import statistics
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Docker Compose project label.")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30)
    return parser.parse_args()


def run_command(cmd: list[str], timeout: float) -> tuple[float, int]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    return time.perf_counter() - started, proc.returncode


def load_containers(project: str, sample: int) -> list[dict]:
    ids = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        text=True,
    ).split()
    if not ids:
        return []
    raw = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.Id}} {{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids,
        text=True,
    )
    containers = []
    for line in raw.splitlines():
        cid, name, pid, running = line.split()[:4]
        if running.lower() == "true" and int(pid) > 0:
            containers.append({"id": cid[:12], "name": name.lstrip("/"), "pid": int(pid)})
    containers.sort(key=lambda item: item["name"])
    return containers[:sample]


def summarize(label: str, wall: float, samples: list[tuple[float, int]]) -> dict:
    durations = sorted(duration for duration, _returncode in samples)
    returncodes = [returncode for _duration, returncode in samples]

    def percentile(q: float) -> float | None:
        if not durations:
            return None
        return durations[min(len(durations) - 1, int(len(durations) * q))]

    return {
        "label": label,
        "count": len(durations),
        "failures": sum(1 for returncode in returncodes if returncode != 0),
        "wall_seconds": round(wall, 6),
        "avg_ms": round(statistics.mean(durations) * 1000, 3) if durations else None,
        "p50_ms": round(percentile(0.50) * 1000, 3) if durations else None,
        "p95_ms": round(percentile(0.95) * 1000, 3) if durations else None,
        "max_ms": round(max(durations) * 1000, 3) if durations else None,
    }


def bench_sequential(label: str, containers: list[dict], make_cmd, timeout: float) -> dict:
    started = time.perf_counter()
    samples = [run_command(make_cmd(container), timeout) for container in containers]
    return summarize(label, time.perf_counter() - started, samples)


def bench_parallel(label: str, containers: list[dict], make_cmd, timeout: float, workers: int) -> dict:
    started = time.perf_counter()
    samples = []
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = [pool.submit(run_command, make_cmd(container), timeout) for container in containers]
        for future in futures.as_completed(pending):
            samples.append(future.result())
    return summarize(label, time.perf_counter() - started, samples)


def main() -> int:
    args = parse_args()
    containers = load_containers(args.project, args.sample)
    print(
        json.dumps(
            {
                "project": args.project,
                "sample": len(containers),
                "parallel": args.parallel,
                "timeout": args.timeout,
            },
            ensure_ascii=False,
        )
    )
    if not containers:
        return 1

    commands = {
        "docker_exec_true": lambda item: ["docker", "exec", item["id"], "true"],
        "nsenter_true": lambda item: ["sudo", "-n", "nsenter", "-t", str(item["pid"]), "-n", "-m", "--", "true"],
        "docker_exec_probe": lambda item: [
            "docker",
            "exec",
            item["id"],
            "sh",
            "-c",
            "command -v bird >/dev/null 2>&1 && test -f /etc/bird/bird.conf",
        ],
        "nsenter_probe": lambda item: [
            "sudo",
            "-n",
            "nsenter",
            "-t",
            str(item["pid"]),
            "-n",
            "-m",
            "--",
            "sh",
            "-c",
            "command -v bird >/dev/null 2>&1 && test -f /etc/bird/bird.conf",
        ],
    }

    for label, make_cmd in commands.items():
        print(json.dumps(bench_sequential(f"{label}_seq", containers, make_cmd, args.timeout), ensure_ascii=False))
    for label, make_cmd in commands.items():
        print(
            json.dumps(
                bench_parallel(f"{label}_par", containers, make_cmd, args.timeout, args.parallel),
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
