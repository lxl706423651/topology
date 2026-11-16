#!/usr/bin/env python3
"""
Consume BIRD protocol-delta events and write them to the current netns main FIB.

The event source is the local BIRD patch in bird-2.17.2-delta/proto/delta.
Run this agent in the router container/netns whose FIB should be updated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RouteEvent:
    ts_us: int
    proto: str
    table: str
    action: str
    prefix: str
    dest: str
    metric: int
    nexthops: list[tuple[str, str, int, int]]
    raw: str

    @property
    def family(self) -> int:
        return 6 if ":" in self.prefix else 4


def parse_route_event(line: str) -> RouteEvent | None:
    if not line or line.startswith("#"):
        return None

    fields = line.rstrip("\n").split("\t")
    if len(fields) < 9 or fields[1] != "R":
        return None

    ts_us = int(fields[0])
    proto = fields[2]
    table = fields[3]
    action = fields[4]
    prefix = fields[5]
    dest = fields[6]
    metric = int(fields[7])
    nh_count = int(fields[8])
    tail = fields[9:]

    nexthops: list[tuple[str, str, int, int]] = []
    if len(tail) < nh_count * 4:
        raise ValueError(f"bad nexthop field count: {line!r}")

    for idx in range(nh_count):
        off = idx * 4
        gw = tail[off]
        iface = tail[off + 1]
        flags = int(tail[off + 2])
        weight = int(tail[off + 3])
        nexthops.append((gw, iface, flags, weight))

    return RouteEvent(ts_us, proto, table, action, prefix, dest, metric, nexthops, line.rstrip("\n"))


def route_command(event: RouteEvent) -> str | None:
    metric = str(event.metric)

    if event.action == "delete":
        # Delete the route key independent of the current BIRD destination type.
        # BIRD may report a delete for an unreachable route after a previous
        # unicast replace for the same prefix; type-specific deletes leave stale
        # routes in that case.
        return f"route del {event.prefix} proto bird metric {metric}"

    if event.action != "replace":
        return None

    if event.dest == "blackhole":
        return f"route replace blackhole {event.prefix} proto bird metric {metric}"
    if event.dest == "unreachable":
        return f"route replace unreachable {event.prefix} proto bird metric {metric}"
    if event.dest in {"prohibited", "prohibit"}:
        return f"route replace prohibit {event.prefix} proto bird metric {metric}"
    if event.dest != "unicast":
        return None

    if not event.nexthops:
        return None

    if len(event.nexthops) == 1:
        gw, iface, flags, _weight = event.nexthops[0]
        parts = ["route", "replace", event.prefix]
        if gw != "-":
            parts += ["via", gw]
        if iface != "-":
            parts += ["dev", iface]
        if flags & 1:
            parts += ["onlink"]
        parts += ["proto", "bird", "metric", metric]
        return " ".join(parts)

    parts = ["route", "replace", event.prefix, "proto", "bird", "metric", metric]
    for gw, iface, flags, weight in event.nexthops:
        parts.append("nexthop")
        if gw != "-":
            parts += ["via", gw]
        if iface != "-":
            parts += ["dev", iface]
        if flags & 1:
            parts.append("onlink")
        parts += ["weight", str(max(1, weight))]
    return " ".join(parts)


def route_order(event: RouteEvent) -> tuple[int, str]:
    if event.action == "delete":
        return (2, event.prefix)

    if event.dest != "unicast":
        return (0, event.prefix)

    if event.nexthops and all(gw == "-" for gw, _iface, _flags, _weight in event.nexthops):
        return (0, event.prefix)

    return (1, event.prefix)


def delete_miss_only(stderr: str) -> bool:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False

    return all(
        line == "RTNETLINK answers: No such process"
        or re.fullmatch(r"Command failed -:\d+", line)
        for line in lines
    )


def run_batch(
    lines: list[str],
    family: int,
    timeout: float,
    dry_run: bool,
    ignore_delete_miss: bool = False,
) -> dict:
    if not lines:
        return {"family": family, "commands": 0, "returncode": 0, "stderr": ""}

    if dry_run:
        return {"family": family, "commands": len(lines), "returncode": 0, "stderr": "dry-run"}

    proc = subprocess.run(
        ["ip", f"-{family}", "-force", "-batch", "-"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    raw_stderr = (proc.stderr or proc.stdout or "").strip()
    ignored_delete_miss = bool(ignore_delete_miss and proc.returncode != 0 and delete_miss_only(raw_stderr))
    stderr = raw_stderr[-4000:]
    return {
        "family": family,
        "commands": len(lines),
        "returncode": 0 if ignored_delete_miss else proc.returncode,
        "stderr": stderr,
        "ignored_delete_miss": ignored_delete_miss,
    }


def flush_pending(pending: dict[str, RouteEvent], args: argparse.Namespace, stats: dict) -> None:
    if not pending:
        return

    events = sorted(pending.values(), key=route_order)
    pending.clear()

    v4_replace_lines: list[str] = []
    v6_replace_lines: list[str] = []
    v4_delete_lines: list[str] = []
    v6_delete_lines: list[str] = []
    skipped = 0

    for event in events:
        cmd = route_command(event)
        if not cmd:
            skipped += 1
            continue
        if event.action == "delete":
            if event.family == 6:
                v6_delete_lines.append(cmd)
            else:
                v4_delete_lines.append(cmd)
        else:
            if event.family == 6:
                v6_replace_lines.append(cmd)
            else:
                v4_replace_lines.append(cmd)

    started = time.monotonic()
    results = [
        run_batch(v4_replace_lines, 4, args.command_timeout, args.dry_run),
        run_batch(v6_replace_lines, 6, args.command_timeout, args.dry_run),
        run_batch(v4_delete_lines, 4, args.command_timeout, args.dry_run, ignore_delete_miss=True),
        run_batch(v6_delete_lines, 6, args.command_timeout, args.dry_run, ignore_delete_miss=True),
    ]
    duration = time.monotonic() - started

    stats["flushes"] += 1
    stats["collapsed_events"] += len(events)
    stats["commands"] += len(v4_replace_lines) + len(v6_replace_lines) + len(v4_delete_lines) + len(v6_delete_lines)
    stats["skipped_events"] += skipped
    stats["write_seconds"] += duration
    stats["ignored_delete_miss_batches"] += sum(1 for item in results if item.get("ignored_delete_miss"))

    errors = [item for item in results if item["returncode"] != 0]
    if errors:
        stats["batch_errors"].extend(errors)
        if not args.keep_going:
            raise RuntimeError(f"ip batch failed: {errors[0]['stderr']}")


def event_lines(
    path: Path,
    from_end: bool,
    follow: bool,
    idle_timeout: float | None,
    since_byte_offset: int = 0,
) -> Iterable[tuple[int, str, int]]:
    start_idle = None
    with path.open("rb") as fh:
        if since_byte_offset > 0:
            fh.seek(since_byte_offset, os.SEEK_SET)
        elif from_end:
            fh.seek(0, os.SEEK_END)

        while True:
            start_pos = fh.tell()
            raw = fh.readline()
            if raw:
                start_idle = None
                yield start_pos, raw.decode("utf-8", errors="replace"), fh.tell()
                continue

            if not follow:
                break

            now = time.monotonic()
            if start_idle is None:
                start_idle = now
            elif idle_timeout is not None and (now - start_idle) >= idle_timeout:
                break
            time.sleep(0.02)


def write_stats(path: Path | None, stats: dict) -> None:
    if not path:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume BIRD delta events and write main FIB batches.")
    parser.add_argument("--event-file", default="/run/bird/delta-events.tsv")
    parser.add_argument("--from-end", action="store_true", help="ignore existing events and follow only new deltas")
    parser.add_argument("--follow", action="store_true", help="continue waiting for new events")
    parser.add_argument("--idle-timeout", type=float, default=None, help="stop after this many idle seconds while following")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--flush-interval", type=float, default=0.05)
    parser.add_argument("--command-timeout", type=float, default=30)
    parser.add_argument("--stats-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--final-collapse",
        action="store_true",
        help="read the selected event range first and write only the final event per prefix",
    )
    parser.add_argument(
        "--since-byte-offset",
        type=int,
        default=0,
        help="start reading the event file at this byte offset; useful for reconvergence deltas",
    )
    parser.add_argument(
        "--write-end-offset-file",
        default=None,
        help="write the final consumed byte offset to this file",
    )
    args = parser.parse_args()

    event_path = Path(args.event_file)
    if not event_path.exists():
        raise SystemExit(f"event file does not exist: {event_path}")

    try:
        file_size = event_path.stat().st_size
    except OSError:
        file_size = 0
    start_byte_offset = min(args.since_byte_offset, file_size) if args.since_byte_offset > 0 else (file_size if args.from_end else 0)

    stats = {
        "event_file": str(event_path),
        "started_at_epoch": time.time(),
        "ended_at_epoch": None,
        "mode": "final-collapse" if args.final_collapse else ("follow" if args.follow else "stream"),
        "final_collapse": args.final_collapse,
        "from_end": args.from_end,
        "follow": args.follow,
        "since_byte_offset": args.since_byte_offset,
        "start_byte_offset": start_byte_offset,
        "end_byte_offset": start_byte_offset,
        "raw_route_events": 0,
        "parse_errors": 0,
        "flushes": 0,
        "collapsed_events": 0,
        "commands": 0,
        "skipped_events": 0,
        "write_seconds": 0.0,
        "read_collapse_seconds": 0.0,
        "batch_errors": [],
        "ignored_delete_miss_batches": 0,
        "dry_run": args.dry_run,
    }
    stats_path = Path(args.stats_file) if args.stats_file else None
    pending: dict[str, RouteEvent] = {}
    last_flush = time.monotonic()
    read_started = time.monotonic()

    try:
        for _start_pos, line, end_pos in event_lines(
            event_path,
            args.from_end,
            args.follow,
            args.idle_timeout,
            args.since_byte_offset,
        ):
            stats["end_byte_offset"] = end_pos
            try:
                event = parse_route_event(line)
            except Exception:
                stats["parse_errors"] += 1
                if not args.keep_going:
                    raise
                continue

            if event is None:
                continue

            stats["raw_route_events"] += 1
            pending[event.prefix] = event

            now = time.monotonic()
            if (
                not args.final_collapse
                and (len(pending) >= args.batch_size or (now - last_flush) >= args.flush_interval)
            ):
                flush_pending(pending, args, stats)
                last_flush = now
                write_stats(stats_path, stats)

            if args.max_events and stats["raw_route_events"] >= args.max_events:
                break

        stats["read_collapse_seconds"] = time.monotonic() - read_started
        stats["pending_events_before_write"] = len(pending)
        flush_pending(pending, args, stats)
        stats["ended_at_epoch"] = time.time()
        if args.write_end_offset_file:
            Path(args.write_end_offset_file).write_text(str(stats["end_byte_offset"]) + "\n", encoding="utf-8")
        write_stats(stats_path, stats)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0 if not stats["batch_errors"] else 1
    except KeyboardInterrupt:
        stats["read_collapse_seconds"] = time.monotonic() - read_started
        flush_pending(pending, args, stats)
        stats["ended_at_epoch"] = time.time()
        write_stats(stats_path, stats)
        return 130


if __name__ == "__main__":
    sys.exit(main())
