#!/usr/bin/env python3
"""Run the K3C initial-FIB workflow, then collect five ICMP capture groups."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys
import time


K3C_ROOT = Path("/home/lxl/0712k3c")
K3C_ENTRY = K3C_ROOT / "run_k3c_local.sh"
K3C_VERIFY = Path(
    "/home/lxl/seed-emulator/topology/bird/packaged/"
    "rtnl_fib_methods_repro_20260701_133858/K3c/verify_bird_fib_vm.py"
)
TOPOLOGY_ROOT = Path(__file__).resolve().parent
OBSERVER_ENTRY = TOPOLOGY_ROOT / "run_traffic_observer_experiment.py"
STAGES = ("k3c", "verify-fib", "observer", "capture")


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def compose_project(output_dir: Path) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", output_dir.name.lower()).strip("-_")
    if not value:
        raise RuntimeError(f"cannot derive Compose project from {output_dir}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_stream(command: list[str], log_path: Path, *, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"command failed rc={returncode}: {' '.join(command)}")


def stage_enabled(start_stage: str, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(start_stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--start-stage", choices=STAGES, default="k3c")
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-timeout", type=float, default=7200.0)
    parser.add_argument("--observer-ready-timeout", type=float, default=7200.0)
    parser.add_argument("--initial-write-jobs", type=int, default=16)
    parser.add_argument("--restart-delay", type=float, default=0.5)
    parser.add_argument("--verify-command-timeout", type=float, default=1800.0)
    parser.add_argument("--min-ping-hops", type=int, default=5)
    parser.add_argument("--max-ping-hops", type=int, default=64)
    parser.add_argument("--pair-probe-timeout", type=int, default=2)
    parser.add_argument("--pair-probe-limit", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.result_dir:
        args.result_dir = args.result_dir.expanduser().resolve()
    if not (args.output_dir / "docker-compose.yml").is_file():
        parser.error(f"missing docker-compose.yml under {args.output_dir}")
    if args.initial_write_jobs < 1:
        parser.error("--initial-write-jobs must be positive")
    if args.restart_delay < 0:
        parser.error("--restart-delay must not be negative")
    for name in (
        "min_ping_hops",
        "max_ping_hops",
        "pair_probe_timeout",
        "pair_probe_limit",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_ping_hops < args.min_ping_hops:
        parser.error("--max-ping-hops must be greater than or equal to --min-ping-hops")
    return args


def main() -> int:
    args = parse_args()
    run_stamp = timestamp()
    project = compose_project(args.output_dir)
    run_name = args.run_name or f"traffic-observer-k3c-{args.output_dir.name}_{run_stamp}"
    result_dir = args.result_dir or (
        TOPOLOGY_ROOT
        / "evaluationResult"
        / "traffic_observer"
        / f"{args.output_dir.name}_k3c_{run_stamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = result_dir / "k3c_pipeline_summary.json"
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    if pipeline_path.is_file() and args.start_stage != "k3c":
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        pipeline.setdefault("resume_history", []).append(
            {"resumed_at": now_utc, "start_stage": args.start_stage}
        )
        pipeline["success"] = False
        pipeline["start_stage"] = args.start_stage
        pipeline["k3c_run_name"] = run_name
        pipeline.pop("error", None)
        pipeline.pop("finished_at", None)
        pipeline.setdefault("stages", {})
    else:
        pipeline = {
            "success": False,
            "started_at": now_utc,
            "output_dir": str(args.output_dir),
            "result_dir": str(result_dir),
            "project_name": project,
            "k3c_run_name": run_name,
            "start_stage": args.start_stage,
            "stages": {},
        }
    write_json(pipeline_path, pipeline)
    try:
        if stage_enabled(args.start_stage, "k3c"):
            started = time.monotonic()
            command = [
                str(K3C_ENTRY),
                str(args.output_dir),
                "--run-name",
                run_name,
                "--project-name",
                project,
                "--stop-after-initial-fib",
                "--initial-write-jobs",
                str(args.initial_write_jobs),
                "--restart-delay",
                str(args.restart_delay),
                "--load-threshold",
                str(args.load_threshold),
                "--max-wait-seconds",
                str(args.load_timeout),
            ]
            run_stream(command, result_dir / "k3c_runner.console.log", cwd=K3C_ROOT)
            k3c_summary_path = K3C_ROOT / "runs" / run_name / "main_summary.json"
            k3c_summary = json.loads(k3c_summary_path.read_text(encoding="utf-8"))
            if not k3c_summary.get("success"):
                raise RuntimeError(f"K3C runner did not succeed: {k3c_summary_path}")
            if k3c_summary.get("execution_scope") != "initial-fib-only":
                raise RuntimeError(
                    "K3C runner did not stop after initial FIB: "
                    f"scope={k3c_summary.get('execution_scope')!r}"
                )
            pipeline["stages"]["k3c"] = {
                "success": True,
                "duration_seconds": round(time.monotonic() - started, 3),
                "summary_path": str(k3c_summary_path),
                "run_dir": str(k3c_summary_path.parent),
            }
            write_json(pipeline_path, pipeline)

        if stage_enabled(args.start_stage, "verify-fib"):
            started = time.monotonic()
            verify_path = result_dir / "verify_after_k3c_fib_all_as.json"
            command = [
                sys.executable,
                str(K3C_VERIFY),
                "--project-name",
                project,
                "--stage",
                "fib-write",
                "--load-threshold",
                str(args.load_threshold),
                "--load-timeout",
                str(args.load_timeout),
                "--poll-interval",
                "30",
                "--command-timeout",
                str(args.verify_command_timeout),
                "--summary-path",
                str(verify_path),
            ]
            run_stream(command, result_dir / "verify_after_k3c_fib.console.log")
            verify = json.loads(verify_path.read_text(encoding="utf-8"))
            if not verify.get("success") or int(verify.get("failure_count") or 0):
                raise RuntimeError(f"all-AS FIB verification failed: {verify_path}")
            pipeline["stages"]["verify_after_fib_write"] = {
                "success": True,
                "duration_seconds": round(time.monotonic() - started, 3),
                "summary_path": str(verify_path),
                "as_count": verify.get("as_count"),
                "load_gate": verify.get("load_gate"),
            }
            write_json(pipeline_path, pipeline)

        observer_start = "capture" if args.start_stage == "capture" else "observer"
        started = time.monotonic()
        command = [
            sys.executable,
            str(OBSERVER_ENTRY),
            "--output-dir",
            str(args.output_dir),
            "--result-dir",
            str(result_dir),
            "--start-stage",
            observer_start,
            "--observer-ready-timeout",
            str(args.observer_ready_timeout),
            "--min-ping-hops",
            str(args.min_ping_hops),
            "--max-ping-hops",
            str(args.max_ping_hops),
            "--pair-probe-timeout",
            str(args.pair_probe_timeout),
            "--pair-probe-limit",
            str(args.pair_probe_limit),
        ]
        run_stream(command, result_dir / "traffic_observer.console.log", cwd=TOPOLOGY_ROOT)
        observer_summary_path = result_dir / "summary.json"
        observer_summary = json.loads(observer_summary_path.read_text(encoding="utf-8"))
        if not observer_summary.get("success"):
            raise RuntimeError(f"traffic observer workflow failed: {observer_summary_path}")
        pipeline["stages"]["traffic_observer"] = {
            "success": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "summary_path": str(observer_summary_path),
        }
        pipeline["success"] = True
        observer_summary["k3c_pipeline_summary"] = str(pipeline_path)
        observer_summary["k3c_run_name"] = run_name
        write_json(observer_summary_path, observer_summary)
        return 0
    except Exception as exc:
        pipeline["error"] = str(exc)
        print(f"K3C traffic-observer pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        pipeline["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        write_json(pipeline_path, pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
