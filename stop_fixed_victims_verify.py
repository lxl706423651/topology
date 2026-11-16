#!/usr/bin/env python3
"""Stop three fixed brdnodes and measure victim-AS RIB reconvergence."""

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROUTER_SERVICE_RE = re.compile(r"^brdnode_(\d+)_r(\d+)$")
VICTIM_SERVICES = (
    "brdnode_1277_r13",
    "brdnode_1269_r22",
    "brdnode_1271_r12",
)


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


def run(command, *, timeout=60, check=True):
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


def running_project_routers(project_name: str):
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
        match = ROUTER_SERVICE_RE.fullmatch(service)
        if match:
            rows.append(
                {
                    "name": name,
                    "service": service,
                    "asn": int(match.group(1)),
                    "rnum": int(match.group(2)),
                }
            )
    if not rows:
        raise ExperimentError(f"no running brdnode containers for project {project_name}")
    return rows


def resolve_victims_and_observers(rows):
    by_service = {row["service"]: row for row in rows}
    missing = [service for service in VICTIM_SERVICES if service not in by_service]
    if missing:
        raise ExperimentError(f"fixed victims are not all running: {missing}")

    victims = [by_service[service] for service in VICTIM_SERVICES]
    grouped = defaultdict(list)
    victim_names = {row["name"] for row in victims}
    for row in rows:
        if row["name"] not in victim_names:
            grouped[row["asn"]].append(row)

    observers = {}
    for victim in victims:
        candidates = sorted(grouped[victim["asn"]], key=lambda row: (row["rnum"], row["name"]))
        if not candidates:
            raise ExperimentError(f"no running non-victim observer for AS group {victim['asn']}")
        observers[victim["asn"]] = candidates[0]
    return victims, observers


def parse_bird_route_count(output: str):
    match = re.search(r"for\s+(\d+)\s+networks", output)
    if match:
        return int(match.group(1)), "networks"
    match = re.search(r"(?:^|\n)\s*(\d+)\s+(?:of\s+\d+\s+)?routes\b", output)
    if match:
        return int(match.group(1)), "routes"
    raise ExperimentError(f"cannot parse birdc show route count output: {output[-1000:]!r}")


def bird_route_count(row):
    proc = run(["docker", "exec", row["name"], "birdc", "show", "route", "count"])
    value, mode = parse_bird_route_count(proc.stdout)
    return value, mode


def take_snapshot(observers):
    counts = {}
    modes = {}
    for asn, row in observers.items():
        value, mode = bird_route_count(row)
        counts[str(asn)] = value
        modes[str(asn)] = mode
    return counts, modes


def wait_for_reconvergence(
    observers,
    baseline,
    interval: float,
    stable_rounds: int,
    timeout: float,
    logger: Logger,
):
    start = time.monotonic()
    previous = None
    changed_asns = set()
    unchanged_streak = 0
    history = []
    required_asns = set(baseline)
    while True:
        counts, modes = take_snapshot(observers)
        changed_asns.update(asn for asn in required_asns if counts[asn] != baseline[asn])
        unchanged = previous is not None and counts == previous
        unchanged_streak = unchanged_streak + 1 if unchanged else 0
        elapsed = time.monotonic() - start
        entry = {
            "elapsed_seconds": elapsed,
            "counts": counts,
            "parse_modes": modes,
            "changed_asns": sorted(changed_asns),
            "unchanged_from_previous": unchanged,
            "unchanged_streak": unchanged_streak,
        }
        history.append(entry)
        logger.log(
            f"RIB sample: elapsed={elapsed:.1f}s, counts={counts}, "
            f"changed_asns={sorted(changed_asns)}, "
            f"unchanged_streak={unchanged_streak}/{stable_rounds}"
        )
        if unchanged_streak >= stable_rounds:
            return elapsed, counts, history
        if elapsed >= timeout:
            raise ExperimentError(f"RIB reconvergence timed out after {elapsed:.1f}s")
        previous = counts
        time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="exportall-1078")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--stop-timeout", type=int, default=10)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--log-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0 or args.stable_rounds != 2:
        raise SystemExit("this experiment requires two unchanged rounds at a positive interval")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path or Path("logs") / f"stop_victims_verify_{timestamp}.log"
    summary_path = args.summary_path or Path("logs") / f"stop_victims_verify_{timestamp}.json"
    logger = Logger(log_path)
    summary = {
        "success": False,
        "project_name": args.project_name,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": vars(args) | {"summary_path": str(summary_path), "log_path": str(log_path)},
    }
    fault_start = None
    try:
        rows = running_project_routers(args.project_name)
        victims, observers = resolve_victims_and_observers(rows)
        summary["victims"] = [
            {"service": row["service"], "container": row["name"], "asn": row["asn"]}
            for row in victims
        ]
        summary["observers"] = {
            str(asn): {"service": row["service"], "container": row["name"]}
            for asn, row in observers.items()
        }
        baseline, modes = take_snapshot(observers)
        summary["baseline"] = {"counts": baseline, "parse_modes": modes}
        logger.log(f"victims: {summary['victims']}")
        logger.log(f"observers: {summary['observers']}")
        logger.log(f"pre-fault RIB baseline: {baseline}")

        fault_start = time.monotonic()
        stop_begin = fault_start
        run(
            [
                "docker",
                "stop",
                "--timeout",
                str(args.stop_timeout),
                *[row["name"] for row in victims],
            ],
            timeout=max(60, args.stop_timeout * len(victims) + 30),
        )
        stop_duration = time.monotonic() - stop_begin
        summary["stop_duration_seconds"] = stop_duration
        logger.log(f"fixed victims stopped in {stop_duration:.3f}s")

        post_stop_elapsed, final_counts, history = wait_for_reconvergence(
            observers, baseline, args.interval, args.stable_rounds, args.timeout, logger
        )
        summary["verification"] = {
            "success": True,
            "criterion": (
                "all three victim-AS bird route counts were unchanged for two consecutive "
                "5-second intervals"
            ),
            "post_stop_convergence_seconds": post_stop_elapsed,
            "history": history,
            "final_counts": final_counts,
        }
        summary["fault_to_convergence_seconds"] = time.monotonic() - fault_start
        summary["success"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        if fault_start is not None:
            summary["fault_to_failure_seconds"] = time.monotonic() - fault_start
        logger.log(f"ERROR: {exc}")
    finally:
        summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.log(
            f"result: success={summary['success']}, "
            f"fault_to_convergence_seconds={summary.get('fault_to_convergence_seconds')}, "
            f"summary={summary_path}"
        )
        logger.close()
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
