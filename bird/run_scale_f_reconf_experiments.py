#!/usr/bin/env python3
"""Run F/reconvergence experiments for several topology scales on the VM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_VM = "192.168.122.60"
DEFAULT_VM_USER = "lxl"
DEFAULT_DOMAIN = "bird-rtnl-dev"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    log("$ " + shlex.join(str(x) for x in cmd))
    stdout = None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = log_file.open("a", encoding="utf-8")
        stdout.write(f"\n# {now()} $ {shlex.join(str(x) for x in cmd)}\n")
        stdout.flush()
    try:
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout else None,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        if stdout:
            stdout.close()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {shlex.join(str(x) for x in cmd)}")
    return proc


def ssh_cmd(vm_user: str, vm: str, remote: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=10",
        f"{vm_user}@{vm}",
        remote,
    ]


def service_count(output_dir: Path) -> int:
    data = yaml.safe_load((output_dir / "docker-compose.yml").read_text(encoding="utf-8")) or {}
    return len(data.get("services", {}))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reset_vm(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    log_file = run_dir / "vm_reset.log"
    result: dict[str, Any] = {"started_at": now(), "success": False}
    run_cmd(["virsh", "destroy", args.domain], log_file=log_file, check=False)
    run_cmd(["virsh", "start", args.domain], log_file=log_file)
    ready = False
    for _ in range(60):
        proc = run_cmd(ssh_cmd(args.vm_user, args.vm, "echo ready"), log_file=log_file, check=False, timeout=8)
        if proc.returncode == 0:
            ready = True
            break
        time.sleep(5)
    if not ready:
        raise RuntimeError("VM SSH did not become ready after virsh start")
    run_cmd(ssh_cmd(args.vm_user, args.vm, "/home/lxl/bird/reset_docker_vm.sh"), log_file=log_file)
    verify = subprocess.run(
        ssh_cmd(
            args.vm_user,
            args.vm,
            "printf 'containers='; docker ps -aq | wc -l; "
            "printf 'volumes='; docker volume ls -q | wc -l; "
            "printf 'custom_networks='; docker network ls --filter type=custom -q | wc -l",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    result.update(
        {
            "success": verify.returncode == 0,
            "duration_seconds": round(time.monotonic() - started, 3),
            "verify": verify.stdout.strip(),
            "ended_at": now(),
            "log_file": str(log_file),
        }
    )
    if verify.returncode != 0:
        result["error"] = verify.stderr.strip()
    return result


def generate_output(scale: int, scale_dir: Path, reuse: bool) -> dict[str, Any]:
    started = time.monotonic()
    output_dir = scale_dir / "output"
    log_file = scale_dir / "generate_output.log"
    if reuse and (output_dir / "docker-compose.yml").is_file():
        return {
            "success": True,
            "skipped": True,
            "output_dir": str(output_dir),
            "service_count": service_count(output_dir),
            "duration_seconds": 0.0,
        }
    if output_dir.exists():
        shutil.rmtree(output_dir)
    default_output = REPO_DIR / "output"
    if default_output.exists():
        backup = scale_dir / f"host_output_backup_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(default_output), str(backup))
    run_cmd([sys.executable, "RR_1897.py", str(scale)], cwd=REPO_DIR, log_file=log_file)
    if not (default_output / "docker-compose.yml").is_file():
        raise RuntimeError(f"RR_1897.py did not produce {default_output / 'docker-compose.yml'}")
    shutil.move(str(default_output), str(output_dir))
    return {
        "success": True,
        "output_dir": str(output_dir),
        "service_count": service_count(output_dir),
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_file": str(log_file),
    }


def run_main_experiment(scale: int, scale_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    summary_path = scale_dir / "main_summary.json"
    log_file = scale_dir / "main_experiment.log"
    project = f"topology-{scale}"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_bird_delta_experiment.py"),
        str(output_dir),
        "--vm",
        args.vm,
        "--vm-user",
        args.vm_user,
        "--project-name",
        project,
        "--build-mode",
        args.build_mode,
        "--defer-delta" if args.defer_delta else "--no-defer-delta",
        "--stable-seconds",
        str(args.stable_seconds),
        "--max-wait-seconds",
        str(args.max_wait_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--stability-samples",
        str(args.stability_samples),
        "--validate-samples",
        "10",
        "--restart-jobs",
        str(args.restart_jobs),
        "--write-jobs",
        str(args.write_jobs),
        "--agent-jobs",
        str(args.agent_jobs),
        "--fib-writer",
        "host-setns",
        "--initial-source",
        args.initial_source,
        "--initial-route-action",
        args.initial_route_action,
        "--summary-path",
        str(summary_path),
    ]
    if args.initial_skip_deletes:
        cmd.append("--initial-skip-deletes")
    else:
        cmd.append("--no-initial-skip-deletes")
    if args.pre_flush_fib:
        cmd.append("--pre-flush-fib")
    run_cmd(cmd, cwd=REPO_DIR, log_file=log_file)
    data = read_json(summary_path)
    return {
        "success": bool(data.get("success")),
        "duration_seconds": round(time.monotonic() - started, 3),
        "summary_path": str(summary_path),
        "log_file": str(log_file),
        "project_name": project,
        "remote_run_dir": data.get("remote_run_dir"),
    }


def run_reconvergence(scale: int, scale_dir: Path, main: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    log_file = scale_dir / "reconvergence.log"
    local_dir = scale_dir / "reconvergence"
    remote_run_dir = str(Path(main["remote_run_dir"]) / "reconvergence")
    env = [
        f"VM_HOST={args.vm}",
        f"VM_USER={args.vm_user}",
        f"PROJECT_NAME={main['project_name']}",
        f"REMOTE_RUN_DIR={remote_run_dir}",
        f"LOCAL_RUN_DIR={local_dir}",
        f"RUN_NAME=reconvergence_scale_{scale}",
        f"DOWN_COUNT={args.down_count}",
        "VALIDATE_SAMPLES=10",
        f"STABLE_SECONDS={args.reconvergence_stable_seconds}",
        f"MAX_WAIT_SECONDS={args.reconvergence_max_wait_seconds}",
        f"POLL_INTERVAL={args.reconvergence_poll_interval}",
        f"STABILITY_SAMPLES={args.stability_samples}",
        f"WRITE_JOBS={args.write_jobs}",
        "INSTALL_JOBS=16",
    ]
    cmd = ["env"] + env + [str(SCRIPT_DIR / "reconvergence_bird_delta.sh"), "run"]
    run_cmd(cmd, cwd=REPO_DIR, log_file=log_file)
    summary_path = local_dir / "summary.json"
    data = read_json(summary_path)
    return {
        "success": bool(data.get("success")),
        "duration_seconds": round(time.monotonic() - started, 3),
        "summary_path": str(summary_path),
        "log_file": str(log_file),
        "remote_run_dir": remote_run_dir,
    }


def pick_stage(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return summary.get("stages", {}).get(name, {})


def render_table(overall: dict[str, Any]) -> str:
    rows = []

    def check_cell(stage: dict[str, Any]) -> str:
        if not stage:
            return "-"
        duration = stage.get("duration_seconds")
        samples = stage.get("samples")
        if isinstance(samples, list):
            total = len(samples)
            ok = sum(1 for item in samples if item.get("success"))
        else:
            total = stage.get("sample_count") or stage.get("samples")
            if isinstance(total, int):
                ok = total if stage.get("success") else 0
            else:
                total = "-"
                ok = "-" if not stage.get("success") else total
        suffix = f" {duration:.3f}s" if isinstance(duration, (int, float)) else ""
        return f"{ok}/{total}{suffix}"

    for item in overall.get("scales", []):
        scale = item["scale"]
        main_path = item.get("main", {}).get("summary_path")
        reconf_path = item.get("reconvergence", {}).get("summary_path")
        main = read_json(Path(main_path)) if main_path and Path(main_path).is_file() else {}
        reconf = read_json(Path(reconf_path)) if reconf_path and Path(reconf_path).is_file() else {}
        stages = main.get("stages", {})
        rstages = reconf.get("stages", {})
        write = pick_stage(main, "write_initial_fib").get("writer_stats", {})
        reconf_write = pick_stage(reconf, "write_delta_fib").get("writer", {})
        rows.append(
            {
                "scale": scale,
                "services": item.get("generate", {}).get("service_count"),
                "build": pick_stage(main, "build").get("duration_seconds"),
                "up": pick_stage(main, "up").get("duration_seconds"),
                "start_bird": pick_stage(main, "restart_bird").get("duration_seconds"),
                "wait_rib": pick_stage(main, "wait_rib_stable").get("duration_seconds"),
                "validate_bird": check_cell(pick_stage(main, "validate_bird")),
                "start_kernel": pick_stage(main, "write_initial_fib").get("duration_seconds"),
                "start_kernel_cmds": write.get("commands"),
                "validate_kernel": check_cell(pick_stage(main, "validate")),
                "reconf_wait": pick_stage(reconf, "wait_rib_stable").get("duration_seconds"),
                "reconf_write": pick_stage(reconf, "write_delta_fib").get("duration_seconds"),
                "reconf_cmds": reconf_write.get("commands"),
                "reconf_validate": check_cell(pick_stage(reconf, "validate")),
                "success": bool(main.get("success")) and bool(reconf.get("success")),
            }
        )
    header = (
        "| Scale | Services | Build(s) | Up(s) | Start-BIRD(s) | RIB stable wait(s) | "
        "BIRD check | Start-kernel/F(s) | F commands | Kernel check | "
        "Reconf wait(s) | Reconf write(s) | Reconf commands | Reconf check | Success |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    lines = [header]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["scale"]),
                    fmt(row["services"]),
                    fmt(row["build"]),
                    fmt(row["up"]),
                    fmt(row["start_bird"]),
                    fmt(row["wait_rib"]),
                    fmt(row["validate_bird"]),
                    fmt(row["start_kernel"]),
                    fmt(row["start_kernel_cmds"]),
                    fmt(row["validate_kernel"]),
                    fmt(row["reconf_wait"]),
                    fmt(row["reconf_write"]),
                    fmt(row["reconf_cmds"]),
                    fmt(row["reconf_validate"]),
                    "yes" if row["success"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def update_markdown(overall: dict[str, Any], path: Path) -> None:
    table = render_table(overall)
    details = []
    for item in overall.get("scales", []):
        details.append(
            f"- `{item['scale']}`: `{item.get('local_dir')}`; "
            f"main `{item.get('main', {}).get('summary_path')}`; "
            f"reconvergence `{item.get('reconvergence', {}).get('summary_path')}`"
        )
    text = f"""# F/Reconvergence Multi-Scale Experiment

Generated at: {now()}

VM: `{overall.get('vm')}`

Run root: `{overall.get('run_dir')}`

## Timing Table

{table}

## Artifacts

{chr(10).join(details)}

## Notes

- `Start-kernel/F` is the host-setns writer stage that writes BIRD-selected routes to main FIB. By default it uses the old F fast path (`birdc show route primary` -> `ip -batch`) for the initial full FIB sync.
- `Reconf write` excludes the RIB/load stable wait and measures only the delta FIB update.
- Each scale records 10-brdnode checks in `validate_bird`, `validate`, and reconvergence `validate` stages.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", type=int, default=[2599, 3083, 4192, 4954])
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--vm-user", default=DEFAULT_VM_USER)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--run-root", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--run-dir", type=Path, help="Resume or continue an existing multiscale run directory.")
    parser.add_argument("--reuse-generated", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--start-at", type=int)
    parser.add_argument("--build-mode", choices=["delta-fast", "fast", "reuse"], default="delta-fast")
    parser.add_argument("--defer-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stable-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-samples", type=int, default=20)
    parser.add_argument("--restart-jobs", type=int, default=2)
    parser.add_argument("--write-jobs", type=int, default=8)
    parser.add_argument("--agent-jobs", type=int, default=16)
    parser.add_argument("--pre-flush-fib", action="store_true")
    parser.add_argument("--initial-source", choices=["birdc-primary", "event"], default="birdc-primary")
    parser.add_argument("--initial-route-action", choices=["replace", "add"], default="replace")
    parser.add_argument("--initial-skip-deletes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--down-count", type=int, default=10)
    parser.add_argument("--reconvergence-stable-seconds", type=int, default=120)
    parser.add_argument("--reconvergence-max-wait-seconds", type=int, default=1800)
    parser.add_argument("--reconvergence-poll-interval", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir if args.run_dir else args.run_root / f"F_reconf_multiscale_{timestamp}"
    overall_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir and overall_path.is_file():
        overall = read_json(overall_path)
        overall["resumed_at"] = now()
        overall["vm"] = args.vm
        overall["run_dir"] = str(run_dir)
        overall.setdefault("scales", [])
        overall["settings"] = vars(args) | {"run_root": str(args.run_root), "run_dir": str(args.run_dir)}
    else:
        overall: dict[str, Any] = {
            "started_at": now(),
            "ended_at": None,
            "success": None,
            "vm": args.vm,
            "run_dir": str(run_dir),
            "settings": vars(args) | {"run_root": str(args.run_root), "run_dir": str(args.run_dir) if args.run_dir else None},
            "scales": [],
        }
    write_json(overall_path, overall)

    selected_scales = args.scales
    if args.start_at is not None:
        selected_scales = [scale for scale in selected_scales if scale >= args.start_at]

    try:
        for scale in selected_scales:
            scale_dir = run_dir / f"scale_{scale}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            item: dict[str, Any] = {"scale": scale, "local_dir": str(scale_dir), "started_at": now(), "success": None}
            overall["scales"].append(item)
            write_json(overall_path, overall)

            log(f"===== scale {scale}: generate output =====")
            item["generate"] = generate_output(scale, scale_dir, args.reuse_generated)
            write_json(overall_path, overall)

            if not args.skip_reset:
                log(f"===== scale {scale}: reset VM =====")
                item["vm_reset_before"] = reset_vm(args, scale_dir)
                write_json(overall_path, overall)

            output_dir = Path(item["generate"]["output_dir"])
            log(f"===== scale {scale}: main experiment =====")
            item["main"] = run_main_experiment(scale, scale_dir, output_dir, args)
            write_json(overall_path, overall)

            log(f"===== scale {scale}: reconvergence =====")
            item["reconvergence"] = run_reconvergence(scale, scale_dir, item["main"], args)
            item["success"] = bool(item["main"].get("success")) and bool(item["reconvergence"].get("success"))
            item["ended_at"] = now()
            write_json(overall_path, overall)
            update_markdown(overall, SCRIPT_DIR / "experiment_F_reconf.md")

            if not item["success"]:
                raise RuntimeError(f"scale {scale} failed")

            if not args.skip_reset and scale != selected_scales[-1]:
                log(f"===== scale {scale}: reset VM after run =====")
                item["vm_reset_after"] = reset_vm(args, scale_dir)
                write_json(overall_path, overall)

        overall["success"] = all(bool(item.get("success")) for item in overall["scales"])
    except Exception as exc:
        overall["success"] = False
        overall["error"] = str(exc)
        log(f"ERROR: {exc}")
    finally:
        overall["ended_at"] = now()
        write_json(overall_path, overall)
        update_markdown(overall, SCRIPT_DIR / "experiment_F_reconf.md")

    return 0 if overall["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
