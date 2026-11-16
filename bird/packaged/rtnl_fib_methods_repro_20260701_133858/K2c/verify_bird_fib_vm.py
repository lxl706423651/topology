#!/usr/bin/env python3
"""Deterministic BIRD/FIB verification for SeedEMU BIRD RTNL experiments.

Run this inside the VM where Docker containers are running.  It enforces the
verification gates described by the seedemu-bird-fib-verify skill:

* after start-bird: wait for load1 < threshold, then check 10 fixed AS samples;
* after FIB write: wait for load1 < threshold, then check one fixed brdnode per AS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import time


ROUTER_RE = re.compile(r"(?:^|[-_])as(\d+)brd-r(\d+)")
BAD_PROTOCOL_RE = re.compile(r"\b(down|failed|error|unreachable)\b", re.IGNORECASE)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def load1() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def wait_load(threshold: float, timeout: float, interval: float) -> dict:
    started = time.monotonic()
    samples = []
    while True:
        value = load1()
        sample = {"at": now_iso(), "load1": value}
        samples.append(sample)
        if value < threshold:
            return {
                "success": True,
                "load1": value,
                "wait_seconds": time.monotonic() - started,
                "samples_tail": samples[-20:],
            }
        if time.monotonic() - started >= timeout:
            return {
                "success": False,
                "load1": value,
                "wait_seconds": time.monotonic() - started,
                "samples_tail": samples[-20:],
            }
        time.sleep(interval)


def docker_brdnodes(project: str) -> list[dict]:
    ps = run_cmd(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"], timeout=120)
    if ps.returncode != 0:
        raise RuntimeError((ps.stderr or ps.stdout or "docker ps failed")[-2000:])
    rows = []
    for line in ps.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        cid, name = parts[0], parts[1]
        match = ROUTER_RE.search(name)
        if not match:
            continue
        rows.append({"id": cid, "name": name, "as": int(match.group(1)), "r": int(match.group(2))})
    rows.sort(key=lambda item: (item["as"], item["r"], item["name"]))
    return rows


def first_per_as(nodes: list[dict]) -> dict[int, dict]:
    reps: dict[int, dict] = {}
    for node in nodes:
        current = reps.get(node["as"])
        if current is None or (node["r"], node["name"]) < (current["r"], current["name"]):
            reps[node["as"]] = node
    return dict(sorted(reps.items()))


def parse_network_count(text: str) -> tuple[int | None, str | None]:
    match = re.search(r"for\s+(\d+)\s+networks?", text)
    if match:
        return int(match.group(1)), "for-networks"
    match = re.search(r"^(\d+)\s+routes?", text.strip())
    if match:
        return int(match.group(1)), "leading-routes"
    match = re.search(r"(\d+)\s+networks?", text)
    if match:
        return int(match.group(1)), "any-networks"
    return None, None


def protocol_bad_lines(text: str) -> list[str]:
    bad = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0].lower() == "name" or parts[1].lower() == "proto":
            continue
        proto = parts[1]
        state = parts[3]
        if proto == "BGP" and "Established" not in line:
            bad.append(line)
            continue
        if state.lower() not in {"up"} and BAD_PROTOCOL_RE.search(line):
            bad.append(line)
    return bad


def verify_after_start_bird(args: argparse.Namespace, nodes: list[dict], load_gate: dict) -> dict:
    reps = first_per_as(nodes)
    selected = [reps[asn] for asn in sorted(reps)[: min(10, len(reps))]]
    samples = []
    for node in selected:
        protocols = run_cmd(["docker", "exec", node["name"], "birdc", "show", "protocols"], timeout=args.command_timeout)
        route_count = run_cmd(["docker", "exec", node["name"], "birdc", "show", "route", "count"], timeout=args.command_timeout)
        route_text = (route_count.stdout or "") + "\n" + (route_count.stderr or "")
        networks, parse_mode = parse_network_count(route_text)
        proto_text = (protocols.stdout or "") + "\n" + (protocols.stderr or "")
        bad_lines = protocol_bad_lines(proto_text) if protocols.returncode == 0 else []
        success = protocols.returncode == 0 and route_count.returncode == 0 and networks is not None and not bad_lines
        samples.append(
            {
                "as": node["as"],
                "name": node["name"],
                "protocol_returncode": protocols.returncode,
                "route_count_returncode": route_count.returncode,
                "bird_networks": networks,
                "parse_mode": parse_mode,
                "bad_protocol_lines": bad_lines,
                "success": success,
                "protocols_tail": proto_text[-2000:] if not success else "",
                "route_count_tail": route_text[-1000:] if not success else "",
            }
        )
    failures = [row for row in samples if not row["success"]]
    return {
        "stage": "verify_after_start_bird",
        "success": load_gate["success"] and bool(samples) and not failures,
        "load_gate": load_gate,
        "sample_count": len(samples),
        "selected_containers": [row["name"] for row in samples],
        "failed_samples": failures,
        "samples": samples,
    }


def verify_after_fib_write(args: argparse.Namespace, nodes: list[dict], load_gate: dict) -> dict:
    reps = first_per_as(nodes)
    samples = []
    for asn, node in reps.items():
        route_count = run_cmd(["docker", "exec", node["name"], "birdc", "show", "route", "count"], timeout=args.command_timeout)
        ip_count = run_cmd(["docker", "exec", node["name"], "sh", "-lc", "ip route | wc -l"], timeout=args.command_timeout)
        route_text = (route_count.stdout or "") + "\n" + (route_count.stderr or "")
        ip_text = (ip_count.stdout or "") + "\n" + (ip_count.stderr or "")
        networks, parse_mode = parse_network_count(route_text)
        try:
            ip_routes = int((ip_count.stdout or "").strip().split()[0])
        except (IndexError, ValueError):
            ip_routes = None
        success = (
            route_count.returncode == 0
            and ip_count.returncode == 0
            and networks is not None
            and ip_routes is not None
            and networks < ip_routes
        )
        samples.append(
            {
                "as": asn,
                "name": node["name"],
                "bird_networks": networks,
                "ip_route_lines": ip_routes,
                "parse_mode": parse_mode,
                "success": success,
                "route_count_returncode": route_count.returncode,
                "ip_count_returncode": ip_count.returncode,
                "route_count_tail": route_text[-1000:] if not success else "",
                "ip_count_tail": ip_text[-1000:] if not success else "",
            }
        )
    failures = [row for row in samples if not row["success"]]
    return {
        "stage": "verify_after_fib_write",
        "success": load_gate["success"] and bool(samples) and not failures,
        "load_gate": load_gate,
        "as_count": len(samples),
        "checked_containers": [row["name"] for row in samples],
        "failed_samples": failures[:50],
        "failure_count": len(failures),
        "samples": samples,
        "condition": "birdc show route count networks < ip route | wc -l",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--stage", choices=["start-bird", "fib-write"], required=True)
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-timeout", type=float, default=7200.0)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    nodes = docker_brdnodes(args.project_name)
    load_gate = wait_load(args.load_threshold, args.load_timeout, args.poll_interval)
    if args.stage == "start-bird":
        result = verify_after_start_bird(args, nodes, load_gate)
    else:
        result = verify_after_fib_write(args, nodes, load_gate)
    result.update(
        {
            "project_name": args.project_name,
            "total_brdnodes": len(nodes),
            "started_at": now_iso(),
            "duration_seconds": time.monotonic() - started,
        }
    )
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
