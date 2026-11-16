#!/usr/bin/env python3
"""Run the stock-BIRD per-netns RTNL baseline inside the experiment VM.

This runner deliberately does not use the F writer or any private bulk route
API.  It separates configuration staging from the timed kernel-protocol
activation so start-kernel measures only serial ``birdc configure`` triggers,
the configured pacing delay, convergence, and correctness verification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


BIRD_NODE_RE = re.compile(r"^as\d+brd-(?:r\d+|ix\d+)-")
EXPORT_ALL_CONFIG = """protocol kernel {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export all;
    };
}
"""


class ExperimentError(RuntimeError):
    pass


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(
    cmd: list[str],
    *,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "command failed").strip()
        raise ExperimentError(f"{' '.join(cmd)}: {message[-2000:]}")
    return proc


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_summary(path: Path, args: argparse.Namespace) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "host": socket.gethostname(),
        "kernel_release": os.uname().release,
        "project_name": args.project_name,
        "settings": {
            "expected_nodes": args.expected_nodes,
            "load_threshold": args.load_threshold,
            "load_stable_seconds": args.load_stable_seconds,
            "stage_timeout": args.stage_timeout,
        },
        "stages": {},
    }


def update_stage(summary_path: Path, summary: dict, name: str, **fields: object) -> None:
    stage = summary.setdefault("stages", {}).setdefault(name, {})
    stage.update(fields)
    summary["updated_at"] = now_iso()
    summary["kernel_release"] = os.uname().release
    write_json(summary_path, summary)


def project_bird_nodes(project_name: str, expected_nodes: int) -> list[str]:
    proc = run_cmd(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--format",
            "{{.Names}}",
        ],
        timeout=120,
        check=True,
    )
    names = sorted(name.strip() for name in proc.stdout.splitlines() if BIRD_NODE_RE.match(name.strip()))
    if len(names) != expected_nodes:
        raise ExperimentError(
            f"expected {expected_nodes} running BIRD nodes for project {project_name}, found {len(names)}"
        )
    return names


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise ExperimentError("stage timeout expired")
    return value


def wait_load_stable(
    *,
    threshold: float,
    stable_seconds: float,
    poll_interval: float,
    deadline: float,
    progress_cb,
) -> dict:
    started = time.monotonic()
    stable_started: float | None = None
    samples: list[dict] = []
    while True:
        current = time.monotonic()
        value = os.getloadavg()[0]
        if value < threshold:
            if stable_started is None:
                stable_started = current
            stable_for = current - stable_started
        else:
            stable_started = None
            stable_for = 0.0
        sample = {
            "at": now_iso(),
            "load1": value,
            "stable_for_seconds": round(stable_for, 3),
        }
        samples.append(sample)
        progress_cb(sample, samples[-120:])
        print(
            f"[{sample['at']}] load1={value:.2f} threshold={threshold:.2f} "
            f"stable={stable_for:.1f}/{stable_seconds:.1f}s",
            flush=True,
        )
        if stable_for >= stable_seconds:
            return {
                "success": True,
                "threshold": threshold,
                "stable_seconds": stable_seconds,
                "wait_seconds": round(current - started, 3),
                "final_load1": value,
                "samples_tail": samples[-120:],
            }
        if current >= deadline:
            raise ExperimentError(
                f"load gate timed out: load1={value:.2f}, stable={stable_for:.1f}/{stable_seconds:.1f}s"
            )
        time.sleep(min(poll_interval, max(0.1, deadline - current)))


def run_verifier(args: argparse.Namespace, run_dir: Path, stage: str, deadline: float) -> dict:
    output = run_dir / f"verify_{stage.replace('-', '_')}.json"
    verify_stage = "start-bird" if stage == "start-bird" else "fib-write"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("verify_bird_fib_vm.py")),
        "--project-name",
        args.project_name,
        "--stage",
        verify_stage,
        "--load-threshold",
        str(args.load_threshold),
        "--load-timeout",
        str(max(1.0, remaining(deadline))),
        "--poll-interval",
        str(args.load_poll_interval),
        "--command-timeout",
        str(args.command_timeout),
        "--summary-path",
        str(output),
    ]
    proc = run_cmd(cmd, timeout=max(1.0, remaining(deadline)))
    if not output.exists():
        raise ExperimentError((proc.stderr or proc.stdout or "verification produced no summary")[-2000:])
    result = json.loads(output.read_text(encoding="utf-8"))
    if proc.returncode != 0 or not result.get("success"):
        raise ExperimentError(f"{stage} verification failed; see {output}")
    return result


def start_bird(args: argparse.Namespace, run_dir: Path, summary_path: Path, summary: dict, export_all: bool) -> None:
    stage_name = "exportall_start_bird" if export_all else "start_bird"
    names = project_bird_nodes(args.project_name, args.expected_nodes)
    started = time.monotonic()
    deadline = started + args.stage_timeout
    failures: list[dict] = []
    update_stage(
        summary_path,
        summary,
        stage_name,
        started_at=now_iso(),
        success=None,
        command=["docker", "exec", "<container>", "bird"],
        jobs=1,
        delay_seconds=args.delay,
        containers_total=len(names),
        processed=0,
        failed=0,
    )
    try:
        dispatch_started = time.monotonic()
        for index, name in enumerate(names, 1):
            proc = run_cmd(
                ["docker", "exec", name, "bird"],
                timeout=min(args.command_timeout, remaining(deadline)),
            )
            if proc.returncode != 0:
                failures.append(
                    {
                        "name": name,
                        "returncode": proc.returncode,
                        "error": (proc.stderr or proc.stdout or "")[-2000:],
                    }
                )
            if index % args.progress_every == 0 or index == len(names):
                elapsed = time.monotonic() - dispatch_started
                print(f"[{now_iso()}] start-bird {index}/{len(names)} elapsed={elapsed:.3f}s", flush=True)
                update_stage(
                    summary_path,
                    summary,
                    stage_name,
                    processed=index,
                    failed=len(failures),
                    dispatch_elapsed_seconds=round(elapsed, 3),
                    failures=failures[:50],
                )
            if index != len(names) and args.delay > 0:
                time.sleep(min(args.delay, remaining(deadline)))
        dispatch_seconds = time.monotonic() - dispatch_started
        if failures:
            raise ExperimentError(f"bird start failed in {len(failures)} containers")

        def load_progress(sample: dict, samples: list[dict]) -> None:
            update_stage(summary_path, summary, stage_name, load_latest=sample, load_samples_tail=samples)

        gate = wait_load_stable(
            threshold=args.load_threshold,
            stable_seconds=args.load_stable_seconds,
            poll_interval=args.load_poll_interval,
            deadline=deadline,
            progress_cb=load_progress,
        )
        verify_started = time.monotonic()
        bird_verify = run_verifier(args, run_dir, "start-bird", deadline)
        fib_verify = run_verifier(args, run_dir, "fib-write", deadline) if export_all else None
        verify_seconds = time.monotonic() - verify_started
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=True,
            dispatch_elapsed_seconds=round(dispatch_seconds, 3),
            fixed_delay_seconds=round(max(0, len(names) - 1) * args.delay, 3),
            load_gate=gate,
            verify_after_start_bird=bird_verify,
            verify_after_fib_write=fib_verify,
            verify_elapsed_seconds=round(verify_seconds, 3),
            duration_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=False,
            error=str(exc),
            failed=len(failures),
            failures=failures[:50],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        raise


def prepare_kernel(args: argparse.Namespace, run_dir: Path, summary_path: Path, summary: dict) -> None:
    stage_name = "prepare_kernel_conf"
    names = project_bird_nodes(args.project_name, args.expected_nodes)
    started = time.monotonic()
    deadline = started + args.stage_timeout
    conf = run_dir / "kernel-export-all.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(EXPORT_ALL_CONFIG, encoding="utf-8")
    failures: list[dict] = []
    completed = 0
    update_stage(
        summary_path,
        summary,
        stage_name,
        started_at=now_iso(),
        success=None,
        source=str(conf),
        destination="/etc/bird/conf/kernel.conf",
        copy_method="docker cp",
        jobs=args.prepare_jobs,
        containers_total=len(names),
        processed=0,
    )

    def copy_one(name: str) -> dict:
        proc = run_cmd(
            ["docker", "cp", str(conf), f"{name}:/etc/bird/conf/kernel.conf"],
            timeout=args.command_timeout,
        )
        return {
            "name": name,
            "success": proc.returncode == 0,
            "error": "" if proc.returncode == 0 else (proc.stderr or proc.stdout or "")[-2000:],
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prepare_jobs)) as pool:
            futures = {pool.submit(copy_one, name): name for name in names}
            for future in concurrent.futures.as_completed(futures):
                remaining(deadline)
                result = future.result()
                completed += 1
                if not result["success"]:
                    failures.append(result)
                if completed % args.progress_every == 0 or completed == len(names):
                    print(f"[{now_iso()}] prepare-kernel {completed}/{len(names)}", flush=True)
                    update_stage(
                        summary_path,
                        summary,
                        stage_name,
                        processed=completed,
                        failed=len(failures),
                        failures=failures[:50],
                    )
        if failures:
            raise ExperimentError(f"kernel.conf copy failed in {len(failures)} containers")

        first = names[0]
        validate_started = time.monotonic()
        validation = run_cmd(
            ["docker", "exec", first, "bird", "-p", "-c", "/etc/bird/bird.conf"],
            timeout=min(args.command_timeout, remaining(deadline)),
        )
        if validation.returncode != 0:
            raise ExperimentError(
                f"kernel.conf validation failed in {first}: "
                f"{(validation.stderr or validation.stdout or '')[-2000:]}"
            )

        def load_progress(sample: dict, samples: list[dict]) -> None:
            update_stage(summary_path, summary, stage_name, load_latest=sample, load_samples_tail=samples)

        gate = wait_load_stable(
            threshold=args.load_threshold,
            stable_seconds=args.load_stable_seconds,
            poll_interval=args.load_poll_interval,
            deadline=deadline,
            progress_cb=load_progress,
        )
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=True,
            processed=completed,
            failed=0,
            validation_container=first,
            validation_command=["docker", "exec", first, "bird", "-p", "-c", "/etc/bird/bird.conf"],
            validation_seconds=round(time.monotonic() - validate_started, 3),
            load_gate=gate,
            duration_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=False,
            error=str(exc),
            processed=completed,
            failed=len(failures),
            failures=failures[:50],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        raise


def start_kernel(args: argparse.Namespace, run_dir: Path, summary_path: Path, summary: dict) -> None:
    stage_name = "start_kernel"
    prepare = summary.get("stages", {}).get("prepare_kernel_conf", {})
    if not prepare.get("success"):
        raise ExperimentError("prepare-kernel must succeed before start-kernel")
    names = project_bird_nodes(args.project_name, args.expected_nodes)
    started = time.monotonic()
    deadline = started + args.stage_timeout
    failures: list[dict] = []
    update_stage(
        summary_path,
        summary,
        stage_name,
        started_at=now_iso(),
        success=None,
        command=["docker", "exec", "<container>", "birdc", "configure"],
        jobs=1,
        delay_seconds=args.delay,
        containers_total=len(names),
        processed=0,
        failed=0,
    )
    try:
        dispatch_started = time.monotonic()
        for index, name in enumerate(names, 1):
            proc = run_cmd(
                ["docker", "exec", name, "birdc", "configure"],
                timeout=min(args.command_timeout, remaining(deadline)),
            )
            if proc.returncode != 0:
                failures.append(
                    {
                        "name": name,
                        "returncode": proc.returncode,
                        "error": (proc.stderr or proc.stdout or "")[-2000:],
                    }
                )
            if index % args.progress_every == 0 or index == len(names):
                elapsed = time.monotonic() - dispatch_started
                print(f"[{now_iso()}] start-kernel {index}/{len(names)} elapsed={elapsed:.3f}s", flush=True)
                update_stage(
                    summary_path,
                    summary,
                    stage_name,
                    processed=index,
                    failed=len(failures),
                    dispatch_elapsed_seconds=round(elapsed, 3),
                    failures=failures[:50],
                )
            if index != len(names) and args.delay > 0:
                time.sleep(min(args.delay, remaining(deadline)))
        dispatch_seconds = time.monotonic() - dispatch_started
        if failures:
            raise ExperimentError(f"birdc configure failed in {len(failures)} containers")

        def load_progress(sample: dict, samples: list[dict]) -> None:
            update_stage(summary_path, summary, stage_name, load_latest=sample, load_samples_tail=samples)

        gate = wait_load_stable(
            threshold=args.load_threshold,
            stable_seconds=args.load_stable_seconds,
            poll_interval=args.load_poll_interval,
            deadline=deadline,
            progress_cb=load_progress,
        )
        verify_started = time.monotonic()
        fib_verify = run_verifier(args, run_dir, "fib-write", deadline)
        verify_seconds = time.monotonic() - verify_started
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=True,
            dispatch_elapsed_seconds=round(dispatch_seconds, 3),
            fixed_delay_seconds=round(max(0, len(names) - 1) * args.delay, 3),
            load_gate=gate,
            verify_after_fib_write=fib_verify,
            verify_elapsed_seconds=round(verify_seconds, 3),
            duration_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        update_stage(
            summary_path,
            summary,
            stage_name,
            ended_at=now_iso(),
            success=False,
            error=str(exc),
            failed=len(failures),
            failures=failures[:50],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=["start-bird", "prepare-kernel", "start-kernel", "exportall-start-bird"],
    )
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, default=3083)
    parser.add_argument("--delay", type=float)
    parser.add_argument("--stage-timeout", type=float, default=3600.0)
    parser.add_argument("--command-timeout", type=float, default=300.0)
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-stable-seconds", type=float, default=120.0)
    parser.add_argument("--load-poll-interval", type=float, default=30.0)
    parser.add_argument("--prepare-jobs", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()
    if args.delay is None:
        args.delay = 0.2 if args.phase == "start-kernel" else 0.1
    if args.delay < 0:
        parser.error("--delay must be >= 0")
    if args.stage_timeout <= 0 or args.command_timeout <= 0:
        parser.error("timeouts must be > 0")
    return args


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    summary = load_summary(summary_path, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    try:
        if args.phase == "start-bird":
            start_bird(args, run_dir, summary_path, summary, export_all=False)
        elif args.phase == "exportall-start-bird":
            start_bird(args, run_dir, summary_path, summary, export_all=True)
        elif args.phase == "prepare-kernel":
            prepare_kernel(args, run_dir, summary_path, summary)
        else:
            start_kernel(args, run_dir, summary_path, summary)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
