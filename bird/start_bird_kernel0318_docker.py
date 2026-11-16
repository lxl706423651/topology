#!/usr/bin/env python3
"""Docker-exec variant of start_bird_kernel0318.py.

This is a kernel-export-only helper. It assumes containers are already up and
BIRD is already running, then updates /etc/bird/conf/kernel.conf one router at
a time using docker exec. It intentionally does not run `birdc reload kernel`;
`birdc configure` is enough to apply the new config and avoids the BIRD CLI
parse issue around the reserved `kernel` token.
"""

import argparse
import datetime as dt
import json
import socket
import sys
from pathlib import Path

from rtnl_experiment import (
    DEFAULT_KERNEL_LOAD_THRESHOLD,
    DEFAULT_KERNEL_PER_ROUTER_DELAY,
    DEFAULT_LOAD_GATE_TIMEOUT,
    DEFAULT_PROJECT,
    docker_router_containers,
    now_iso,
    order_by_as_round_robin,
    run_legacy0318_kernel_export,
    write_json,
)


def default_work_dir():
    return Path.home() / "bird" / "runs" / f"start-kernel0318_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--work-dir", help="Default: ~/bird/runs/start-kernel0318_<timestamp>")
    parser.add_argument("--limit-containers", type=int)
    parser.add_argument("--kernel-load-threshold", type=float, default=DEFAULT_KERNEL_LOAD_THRESHOLD)
    parser.add_argument("--kernel-load-timeout", type=float, default=DEFAULT_LOAD_GATE_TIMEOUT)
    parser.add_argument("--kernel-per-router-delay", type=float, default=DEFAULT_KERNEL_PER_ROUTER_DELAY)
    parser.add_argument("--load-check-interval", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=180.0)
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser() if args.work_dir else default_work_dir()
    summary_path = work_dir / "summary.json"
    containers = docker_router_containers(args.project, args.limit_containers)
    ordered = order_by_as_round_robin(containers)
    summary = {
        "schema_version": 1,
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "error": None,
        "host": socket.gethostname(),
        "command": sys.argv,
        "project": args.project,
        "settings": vars(args),
        "containers": {
            "routers": len(containers),
            "first": containers[0]["name"] if containers else None,
            "limit": args.limit_containers,
        },
        "results": {"A": {"description": "standalone legacy0318 docker-exec kernel export", "stages": {}}},
    }
    write_json(summary_path, summary)
    try:
        run_legacy0318_kernel_export(ordered, args, summary["results"]["A"], summary, summary_path)
        summary["success"] = True
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        summary["ended_at"] = now_iso()
        write_json(summary_path, summary)
    print(f"Summary: {summary_path}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
