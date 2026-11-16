#!/usr/bin/env python3
"""Start BIRD for one Compose project and measure FIB convergence.

This keeps the paced nsenter-based startup used by start_bird0318.py, but
enters the container PID namespace, limits discovery to one Compose project,
and adds deterministic convergence sampling after all BIRD processes have
been triggered.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROUTER_SERVICE_RE = re.compile(r"^brdnode_(\d+)_r(\d+)$")


class ExperimentError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run(command, *, timeout=30, check=True):
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise ExperimentError(
            f"command failed rc={proc.returncode}: {' '.join(command)}\n"
            f"stdout={proc.stdout[-1000:]}\nstderr={proc.stderr[-1000:]}"
        )
    return proc


def project_containers(project_name: str):
    proc = run(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            '{{.Names}}\t{{.Label "com.docker.compose.service"}}',
        ]
    )
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, service = line.split("\t", 1)
        if not name.startswith("as"):
            continue
        pid = run(["docker", "inspect", "--format", "{{.State.Pid}}", name]).stdout.strip()
        if not pid.isdigit() or int(pid) <= 0:
            raise ExperimentError(f"container has no running PID: {name}")
        rows.append({"name": name, "service": service, "pid": pid})
    if not rows:
        raise ExperimentError(f"no running BIRD containers found for project {project_name}")
    return rows


def nsenter(row, command, *, timeout=30, check=True):
    return run(
        ["nsenter", "-t", row["pid"], "-n", "-m", *command],
        timeout=timeout,
        check=check,
    )


def first_router_per_as(rows):
    grouped = defaultdict(list)
    for row in rows:
        match = ROUTER_SERVICE_RE.fullmatch(row["service"])
        if match:
            grouped[int(match.group(1))].append((int(match.group(2)), row["name"], row))
    if len(grouped) < 5:
        raise ExperimentError(f"need at least 5 router AS groups, found {len(grouped)}")
    return {
        asn: sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]
        for asn, candidates in grouped.items()
    }


def start_birds(rows, logger: Logger, start_delay: float):
    started = 0
    already_running = 0
    failures = []
    pending = []
    for index, row in enumerate(sorted(rows, key=lambda item: (item["service"], item["name"])), 1):
        present = nsenter(row, ["pgrep", "-x", "bird"], check=False)
        if present.returncode == 0:
            already_running += 1
            continue
        proc = subprocess.Popen(
            ["nsenter", "-t", row["pid"], "-p", "-n", "-m", "bird"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        pending.append((row, proc))
        started += 1
        if started % 200 == 0:
            logger.log(f"start progress: triggered={started}, scanned={index}/{len(rows)}")
            time.sleep(5)
        time.sleep(start_delay)

    for row, proc in pending:
        try:
            _, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
            failures.append({"container": row["name"], "error": "bird start timed out"})
            continue
        if proc.returncode != 0:
            failures.append(
                {
                    "container": row["name"],
                    "returncode": proc.returncode,
                    "stderr": stderr[-1000:],
                }
            )

    logger.log(
        f"BIRD start finished: targets={len(rows)}, started={started}, "
        f"already_running={already_running}, failures={len(failures)}"
    )
    if failures:
        raise ExperimentError(f"BIRD failed to start in {len(failures)} containers: {failures[:5]}")
    return {"targets": len(rows), "started": started, "already_running": already_running}


def wait_for_load(threshold: float, interval: float, timeout: float, logger: Logger):
    start = time.monotonic()
    while True:
        load1 = os.getloadavg()[0]
        elapsed = time.monotonic() - start
        logger.log(f"load gate: load1={load1:.2f}, target=<{threshold}, waited={elapsed:.1f}s")
        if load1 < threshold:
            return load1, elapsed
        if elapsed >= timeout:
            raise ExperimentError(f"load1 remained >= {threshold} for {elapsed:.1f}s")
        time.sleep(interval)


def ip_route_count(row):
    proc = nsenter(row, ["sh", "-lc", "ip route | wc -l"], timeout=60)
    value = proc.stdout.strip()
    if not value.isdigit():
        raise ExperimentError(f"invalid ip route count from {row['name']}: {value!r}")
    return int(value)


def protocol_health(row):
    protocols = nsenter(row, ["birdc", "show", "protocols"], timeout=60)
    route_count = nsenter(row, ["birdc", "show", "route", "count"], timeout=60)
    failures = []
    parsed = 0
    for line in protocols.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] in {"BIRD", "Name"}:
            continue
        parsed += 1
        proto_type = parts[1].lower()
        state = parts[3].lower()
        if state != "up":
            failures.append(line)
        elif proto_type == "bgp" and "Established" not in line:
            failures.append(line)
    if parsed == 0:
        failures.append("no protocol rows parsed")
    return {
        "success": not failures,
        "failed_protocol_lines": failures[:50],
        "protocol_output": protocols.stdout if failures else "",
        "route_count_output": route_count.stdout.strip(),
    }


def wait_for_fib_stability(observers, interval: float, stable_rounds: int, timeout: float, logger: Logger):
    start = time.monotonic()
    previous = None
    unchanged_streak = 0
    history = []
    while True:
        counts = {str(asn): ip_route_count(row) for asn, row in observers.items()}
        elapsed = time.monotonic() - start
        unchanged = previous is not None and counts == previous
        unchanged_streak = unchanged_streak + 1 if unchanged else 0
        entry = {
            "elapsed_seconds": elapsed,
            "counts": counts,
            "unchanged_from_previous": unchanged,
            "unchanged_streak": unchanged_streak,
        }
        history.append(entry)
        logger.log(
            f"FIB sample: elapsed={elapsed:.1f}s, counts={counts}, "
            f"unchanged_streak={unchanged_streak}/{stable_rounds}"
        )
        if unchanged_streak >= stable_rounds:
            health = {str(asn): protocol_health(row) for asn, row in observers.items()}
            unhealthy = [asn for asn, result in health.items() if not result["success"]]
            if not unhealthy:
                return elapsed, counts, history, health
            logger.log(f"protocol verification not ready for AS groups: {unhealthy}; continue waiting")
            unchanged_streak = 0
        if elapsed >= timeout:
            raise ExperimentError(f"FIB convergence timed out after {elapsed:.1f}s")
        previous = counts
        time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="exportall-1078")
    parser.add_argument("--sample-as-count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-check-interval", type=float, default=30.0)
    parser.add_argument("--start-delay", type=float, default=0.02)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--log-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit("must run as root (use sudo -n) for nsenter")
    if args.sample_as_count != 5:
        raise SystemExit("this experiment requires exactly 5 AS samples")
    if args.interval <= 0 or args.stable_rounds != 2:
        raise SystemExit("this experiment requires two unchanged rounds at a positive interval")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path or Path("logs") / f"start_bird_verify_{timestamp}.log"
    summary_path = args.summary_path or Path("logs") / f"start_bird_verify_{timestamp}.json"
    logger = Logger(log_path)
    total_start = time.monotonic()
    summary = {
        "success": False,
        "project_name": args.project_name,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": vars(args) | {"summary_path": str(summary_path), "log_path": str(log_path)},
    }
    try:
        rows = project_containers(args.project_name)
        representatives = first_router_per_as(rows)
        selected_asns = sorted(representatives)[: args.sample_as_count]
        observers = {asn: representatives[asn] for asn in selected_asns}
        summary["selected_containers"] = {
            str(asn): {"service": row["service"], "container": row["name"]}
            for asn, row in observers.items()
        }
        logger.log(f"selected deterministic observers: {summary['selected_containers']}")

        start_begin = time.monotonic()
        summary["start_bird"] = start_birds(rows, logger, args.start_delay)
        summary["start_bird"]["duration_seconds"] = time.monotonic() - start_begin

        load1, load_wait = wait_for_load(
            args.load_threshold, args.load_check_interval, args.timeout, logger
        )
        summary["load_gate"] = {"load1": load1, "wait_seconds": load_wait}

        stable_elapsed, counts, history, health = wait_for_fib_stability(
            observers, args.interval, args.stable_rounds, args.timeout, logger
        )
        summary["verification"] = {
            "success": True,
            "criterion": "all 5 ip-route counts unchanged for two consecutive 5-second intervals",
            "duration_seconds": stable_elapsed,
            "final_counts": counts,
            "history": history,
            "protocol_health": health,
        }
        summary["success"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        logger.log(f"ERROR: {exc}")
    finally:
        summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        summary["total_convergence_seconds"] = time.monotonic() - total_start
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.log(
            f"result: success={summary['success']}, "
            f"total_convergence_seconds={summary['total_convergence_seconds']:.3f}, "
            f"summary={summary_path}"
        )
        logger.close()
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
