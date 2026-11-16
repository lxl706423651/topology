#!/usr/bin/env python3
"""Build and start a SeedEMU Docker output directory.

This runner intentionally stops at Docker image build and container startup.
BIRD startup, kernel export, RTNL pacing, and convergence checks belong to the
separate RTNL experiment runners.
"""

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_all as core


DEFAULT_BUILD_MODE = "fast"
DEFAULT_BUILD_BATCH = 50
DEFAULT_BUILD_JOBS = 16
DEFAULT_UP_BATCH = 50
DEFAULT_UP_JOBS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="SeedEMU compiler output directory containing docker-compose.yml.")
    parser.add_argument("--project-name", help="Docker Compose project name.")
    parser.add_argument(
        "--build-mode",
        choices=["fast", "compose"],
        default=DEFAULT_BUILD_MODE,
        help="Use fast image archive build or docker compose build batches.",
    )
    parser.add_argument("--build-batch", type=int, default=DEFAULT_BUILD_BATCH)
    parser.add_argument("--build-jobs", type=int, default=DEFAULT_BUILD_JOBS)
    parser.add_argument("--up-batch", type=int, default=DEFAULT_UP_BATCH)
    parser.add_argument("--up-jobs", type=int, default=DEFAULT_UP_JOBS)
    parser.add_argument("--service-limit", type=int, help="Only build/up the first N active services. For smoke tests.")
    parser.add_argument("--down-first", action="store_true", help="Run docker compose down before build/up.")
    parser.add_argument("--no-build", action="store_true", help="Skip build and reuse existing images.")
    parser.add_argument("--reuse-images", action="store_true", help="Alias for --no-build.")
    parser.add_argument("--no-up", action="store_true", help="Skip docker compose up.")
    parser.add_argument("--pull", action="store_true", help="Accepted for compatibility; compose build only.")
    parser.add_argument("--summary-path", help="Path for summary.json. Default: <output_dir>/summary.json.")
    parser.add_argument("--vm", help="Sync output_dir to this VM over SSH, then run this script on the VM.")
    parser.add_argument("--vm-user", default="lxl", help="SSH user for --vm mode.")
    parser.add_argument(
        "--vm-output-root",
        help="Remote VM run root. Default: /home/<vm-user>/bird/runs.",
    )
    parser.add_argument("--vm-runner", help="Remote path for this script. Default: /home/<vm-user>/bird/run.py.")
    return parser.parse_args()


def active_services(base_cmd, cwd, logger, service_limit=None):
    services = core.get_services(base_cmd, cwd, logger)
    active, skipped = core.filter_build_up_services(services, logger)
    if service_limit is not None:
        if service_limit <= 0:
            raise ValueError("--service-limit must be > 0")
        skipped += active[service_limit:]
        active = active[:service_limit]
    if not active:
        raise core.StageError("no services left after applying build/up filters", {"services_discovered": len(services)})
    return services, active, skipped


def run_compose_build_subset(base_cmd, output_dir, cwd, logger, services, batch_size, parallel_jobs, summary, pull=False):
    started = time.monotonic()
    log_file = core.script_log_dir(cwd) / f"dockerBuild_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    core.write_log_header(log_file)
    info = {
        "mode": "compose_subset",
        "log_file": str(log_file),
        "batch_size": batch_size,
        "parallel_jobs": parallel_jobs,
        "services_total": len(services),
        "batches_total": 0,
        "batches_succeeded": 0,
        "batches_failed": 0,
        "failed_batches": [],
    }
    summary.update_stage("build", **info)
    logger.log("Checking build context size (target: output directory)...")
    core.run_logged(["du", "-sh", str(output_dir)], cwd, logger)

    warmup_service = services[0]
    info["warmup_service"] = warmup_service
    info["warmup_success"] = None
    summary.update_stage("build", **info)
    logger.log(f"Building warmup service first: {warmup_service}")

    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1"
    build_prefix = base_cmd + ["build"]
    if pull:
        build_prefix.append("--pull")

    with log_file.open("a", encoding="utf-8") as fh:
        warmup = subprocess.run(
            build_prefix + [warmup_service],
            cwd=cwd,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    info["warmup_success"] = warmup.returncode == 0
    summary.update_stage("build", **info)
    if warmup.returncode != 0:
        raise core.StageError(f"warmup build failed for {warmup_service}", info)

    remaining_services = services[1:]
    if not remaining_services:
        info["batches_total"] = 0
        info["duration_seconds"] = int(time.monotonic() - started)
        return info

    batches = list(core.chunks(remaining_services, batch_size))
    info["batches_total"] = len(batches)
    summary.update_stage("build", **info)
    logger.log(
        "Starting docker compose build batches: "
        f"remaining_services={len(remaining_services)} batches={len(batches)} jobs={parallel_jobs}"
    )

    def build_batch(batch):
        batch_started = time.monotonic()
        batch_id = hashlib.md5(" ".join(batch).encode("utf-8")).hexdigest()[:5]
        with log_file.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(
                build_prefix + batch,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return {
            "batch_id": batch_id,
            "returncode": proc.returncode,
            "duration_seconds": int(time.monotonic() - batch_started),
        }

    with ThreadPoolExecutor(max_workers=max(1, parallel_jobs)) as pool:
        futures = [pool.submit(build_batch, batch) for batch in batches]
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] == 0:
                info["batches_succeeded"] += 1
            else:
                info["batches_failed"] += 1
                info["failed_batches"].append(result["batch_id"])
            summary.update_stage("build", **info)

    info["duration_seconds"] = int(time.monotonic() - started)
    logger.log(f"Build stage finished in {info['duration_seconds']}s; log={log_file}")
    if info["batches_failed"]:
        raise core.StageError(f"docker compose build failed for {info['batches_failed']} batch(es)", info)
    return info


def run_compose_up_subset(base_cmd, cwd, logger, services, batch_size, parallel_jobs, summary):
    started = time.monotonic()
    log_file = core.script_log_dir(cwd) / f"dockerUp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    core.write_log_header(log_file)
    batches = list(core.chunks(services, batch_size))
    info = {
        "mode": "compose_subset",
        "log_file": str(log_file),
        "batch_size": batch_size,
        "parallel_jobs": parallel_jobs,
        "services_total": len(services),
        "batches_total": len(batches),
        "batches_succeeded": 0,
        "batches_failed": 0,
        "failed_batches": [],
    }
    summary.update_stage("up", **info)
    logger.log(f"Starting docker compose up batches: services={len(services)} batches={len(batches)} jobs={parallel_jobs}")

    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "0"

    def up_batch(batch):
        batch_started = time.monotonic()
        batch_id = hashlib.md5(" ".join(batch).encode("utf-8")).hexdigest()[:5]
        with log_file.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(
                base_cmd + ["up", "-d"] + batch,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return {
            "batch_id": batch_id,
            "returncode": proc.returncode,
            "duration_seconds": int(time.monotonic() - batch_started),
        }

    with ThreadPoolExecutor(max_workers=max(1, parallel_jobs)) as pool:
        futures = [pool.submit(up_batch, batch) for batch in batches]
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] == 0:
                info["batches_succeeded"] += 1
            else:
                info["batches_failed"] += 1
                info["failed_batches"].append(result["batch_id"])
            summary.update_stage("up", **info)

    info["duration_seconds"] = int(time.monotonic() - started)
    logger.log(f"Up stage finished in {info['duration_seconds']}s; log={log_file}")
    if info["batches_failed"]:
        raise core.StageError(f"docker compose up failed for {info['batches_failed']} batch(es)", info)
    return info


def run_stage(summary, name, func, *args, **kwargs):
    summary.start_stage(name)
    try:
        info = func(*args, **kwargs)
        summary.finish_stage(name, True, **(info or {}))
        return info
    except core.StageError as exc:
        summary.finish_stage(name, False, error=str(exc), **exc.details)
        raise
    except Exception as exc:
        summary.finish_stage(name, False, error=str(exc))
        raise


def local_summary_path(output_dir, raw):
    if raw:
        return Path(raw).expanduser().resolve()
    return output_dir / "summary.json"


def run_local(args, output_dir, summary_path):
    args.no_build = args.no_build or args.reuse_images
    project_name = args.project_name or core.default_project_name(output_dir)
    run_cwd = output_dir.parent
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / "run_logs" / f"seedemu_run_{timestamp}.log"
    logger = core.Logger(log_path)
    summary = core.SummaryRecorder(
        summary_path,
        {
            "mode": "local-build-up",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "output_dir": str(output_dir),
            "project_name": project_name,
            "log_file": str(log_path),
            "settings": {
                "build_mode": args.build_mode,
                "build_batch": args.build_batch,
                "build_jobs": args.build_jobs,
                "up_batch": args.up_batch,
                "up_jobs": args.up_jobs,
                "service_limit": args.service_limit,
                "down_first": args.down_first,
                "no_build": args.no_build,
                "no_up": args.no_up,
            },
        },
    )

    try:
        logger.log("SeedEMU Docker build/up run started.")
        logger.log(f"Output directory: {output_dir}")
        logger.log(f"Project name: {project_name}")
        logger.log(f"Summary: {summary_path}")

        base_cmd = core.compose_cmd(output_dir, project_name)
        all_services, selected, skipped = active_services(base_cmd, run_cwd, logger, args.service_limit)
        summary.update(
            services={
                "discovered": len(all_services),
                "selected": len(selected),
                "skipped": len(skipped),
                "skipped_services": skipped[:50],
            }
        )
        logger.log(f"Compose services: discovered={len(all_services)} selected={len(selected)} skipped={len(skipped)}")

        if args.down_first:
            run_stage(summary, "down", lambda: core.run_logged(base_cmd + ["down", "--remove-orphans"], run_cwd, logger) or {})

        if args.no_build:
            summary.skip_stage("build", "--no-build/--reuse-images", services_total=len(selected))
        elif args.build_mode == "fast" and args.service_limit is None:
            run_stage(summary, "build", core.run_fastdocker_flow, base_cmd, output_dir, run_cwd, logger, project_name, summary)
        else:
            run_stage(
                summary,
                "build",
                run_compose_build_subset,
                base_cmd,
                output_dir,
                run_cwd,
                logger,
                selected,
                args.build_batch,
                args.build_jobs,
                summary,
                args.pull,
            )

        if args.no_up:
            summary.skip_stage("up", "--no-up", services_total=len(selected))
        elif args.service_limit is None:
            run_stage(summary, "up", core.run_dockerup_flow, base_cmd, run_cwd, logger, args.up_batch, args.up_jobs, summary)
        else:
            run_stage(summary, "up", run_compose_up_subset, base_cmd, run_cwd, logger, selected, args.up_batch, args.up_jobs, summary)

        core.run_logged(base_cmd + ["ps"], run_cwd, logger)
        logger.log("SeedEMU Docker build/up run completed successfully.")
        summary.finalize(True)
        return 0
    except Exception as exc:
        logger.log(f"ERROR: {exc}")
        summary.finalize(False, error=str(exc))
        return 1
    finally:
        logger.close()


def ssh_base(vm_user, vm_host):
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
        f"{vm_user}@{vm_host}",
    ]


def rsync_base(vm_user, vm_host):
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    return ["rsync", "-a", "--delete", "-e", ssh_cmd]


def run_streamed(cmd, cwd=None, check=True):
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")
    return proc.returncode


def run_with_input(cmd, payload):
    print(f"$ {shlex.join(cmd)} < input", flush=True)
    proc = subprocess.run(cmd, input=payload, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")


def remote_run_paths(output_dir, project_name, vm_user, vm_output_root, timestamp=None):
    root = Path(vm_output_root or f"/home/{vm_user}/bird/runs")
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    services = core.compose_service_names_from_file(output_dir)
    service_count = len(services) if services else "unknown"
    project_slug = core.slugify(project_name, "project")
    run_dir = root / f"topology-{service_count}_{timestamp}_{project_slug}"
    return run_dir, run_dir / "output", service_count


def remote_command_args(args, remote_runner, remote_output, remote_summary):
    cmd = [str(remote_runner), str(remote_output)]
    if args.project_name:
        cmd += ["--project-name", args.project_name]
    cmd += ["--build-mode", args.build_mode]
    cmd += ["--build-batch", str(args.build_batch)]
    cmd += ["--build-jobs", str(args.build_jobs)]
    cmd += ["--up-batch", str(args.up_batch)]
    cmd += ["--up-jobs", str(args.up_jobs)]
    cmd += ["--summary-path", str(remote_summary)]
    if args.service_limit is not None:
        cmd += ["--service-limit", str(args.service_limit)]
    if args.down_first:
        cmd.append("--down-first")
    if args.no_build or args.reuse_images:
        cmd.append("--no-build")
    if args.no_up:
        cmd.append("--no-up")
    if args.pull:
        cmd.append("--pull")
    return cmd


def fetch_remote_summary(vm_user, vm_host, remote_summary, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    test_cmd = ssh_base(vm_user, vm_host) + ["test", "-f", str(remote_summary)]
    for attempt in range(1, 4):
        if run_streamed(test_cmd, check=False) != 0:
            if attempt < 3:
                time.sleep(5)
                continue
            return False
        rc = run_streamed(
            rsync_base(vm_user, vm_host) + [f"{vm_user}@{vm_host}:{remote_summary}", str(local_path)],
            check=False,
        )
        if rc == 0:
            return True
        if attempt < 3:
            time.sleep(5)
    return False


def attach_vm_host_summary(local_summary_path, host_data):
    if local_summary_path.exists():
        data = json.loads(local_summary_path.read_text(encoding="utf-8"))
    else:
        data = host_data
    data["vm_host"] = {
        "started_at": host_data.get("started_at"),
        "ended_at": host_data.get("ended_at"),
        "duration_seconds": host_data.get("duration_seconds"),
        "success": host_data.get("success"),
        "error": host_data.get("error"),
        "stages": host_data.get("stages", {}),
    }
    tmp = local_summary_path.with_suffix(local_summary_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(local_summary_path)


def run_on_vm(args, output_dir, requested_summary_path=None):
    args.no_build = args.no_build or args.reuse_images
    vm_host = args.vm
    vm_user = args.vm_user
    if not args.project_name:
        args.project_name = core.default_project_name(output_dir)
    stable_runner = Path(args.vm_runner or f"/home/{vm_user}/bird/run.py")
    stable_run_all = stable_runner.parent / "run_all.py"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_dir, remote_output, topology_size = remote_run_paths(
        output_dir,
        args.project_name,
        vm_user,
        args.vm_output_root,
        timestamp,
    )
    local_run_dir = Path(__file__).resolve().parent / "runs" / f"topology-{topology_size}_{timestamp}"
    local_summary_path = requested_summary_path or (local_run_dir / "summary.json")
    compatibility_summary_path = output_dir / "summary.json"
    local_remote_summary_path = local_summary_path.parent / "remote_summary.json"
    remote_runner = remote_run_dir / stable_runner.name
    remote_summary = remote_run_dir / "summary.json"
    remote_manifest = remote_run_dir / "run_manifest.json"

    summary = core.SummaryRecorder(
        local_summary_path,
        {
            "mode": "vm-host-build-up",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "local_output_dir": str(output_dir),
            "local_run_dir": str(local_summary_path.parent),
            "local_summary_path": str(local_summary_path),
            "compatibility_summary_path": str(compatibility_summary_path),
            "remote_run_dir": str(remote_run_dir),
            "remote_output_dir": str(remote_output),
            "remote_summary_path": str(remote_summary),
            "local_remote_summary_path": str(local_remote_summary_path),
            "vm": {"host": vm_host, "user": vm_user, "runner": str(remote_runner), "stable_runner": str(stable_runner)},
            "project_name": args.project_name,
            "topology_services": topology_size,
            "settings": {
                "build_mode": args.build_mode,
                "build_batch": args.build_batch,
                "build_jobs": args.build_jobs,
                "up_batch": args.up_batch,
                "up_jobs": args.up_jobs,
                "service_limit": args.service_limit,
                "no_build": args.no_build,
                "no_up": args.no_up,
            },
        },
    )

    print("SeedEMU run.py VM build/up mode started.", flush=True)
    print(f"Remote run dir: {remote_run_dir}", flush=True)
    print(f"Local summary: {local_summary_path}", flush=True)

    remote_error = None
    try:
        summary.start_stage("sync", remote_run_dir=str(remote_run_dir), remote_output_dir=str(remote_output))
        run_streamed(ssh_base(vm_user, vm_host) + ["mkdir", "-p", str(remote_run_dir), str(remote_output), str(stable_runner.parent)])
        run_streamed(rsync_base(vm_user, vm_host) + [str(Path(__file__).resolve()), f"{vm_user}@{vm_host}:{stable_runner}"])
        run_streamed(rsync_base(vm_user, vm_host) + [str(Path(__file__).resolve()), f"{vm_user}@{vm_host}:{remote_runner}"])
        run_streamed(rsync_base(vm_user, vm_host) + [str(Path(core.__file__).resolve()), f"{vm_user}@{vm_host}:{stable_run_all}"])
        run_streamed(rsync_base(vm_user, vm_host) + [str(Path(core.__file__).resolve()), f"{vm_user}@{vm_host}:{remote_run_dir / 'run_all.py'}"])
        helper_path = None
        if args.build_mode == "fast" and not args.no_build and args.service_limit is None:
            helper_path = core.resolve_fast_builder_script()
            run_streamed(rsync_base(vm_user, vm_host) + [str(helper_path), f"{vm_user}@{vm_host}:{stable_runner.parent / helper_path.name}"])
            run_streamed(rsync_base(vm_user, vm_host) + [str(helper_path), f"{vm_user}@{vm_host}:{remote_run_dir / helper_path.name}"])
        chmod_targets = [str(stable_runner), str(remote_runner), str(stable_run_all), str(remote_run_dir / "run_all.py")]
        if helper_path:
            chmod_targets.extend([str(stable_runner.parent / helper_path.name), str(remote_run_dir / helper_path.name)])
        run_streamed(ssh_base(vm_user, vm_host) + ["chmod", "+x"] + chmod_targets)
        manifest = {
            "schema_version": 1,
            "created_at": core.now_iso(),
            "mode": "run.py vm-host-build-up",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "local_output_dir": str(output_dir),
            "remote_run_dir": str(remote_run_dir),
            "remote_output_dir": str(remote_output),
            "remote_summary_path": str(remote_summary),
            "remote_runner": str(remote_runner),
            "stable_runner": str(stable_runner),
            "stable_run_all": str(stable_run_all),
            "fast_builder_helper": str(helper_path) if helper_path else None,
            "project_name": args.project_name,
            "topology_services": topology_size,
        }
        run_with_input(
            ssh_base(vm_user, vm_host) + [f"cat > {shlex.quote(str(remote_manifest))}"],
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        run_streamed(
            rsync_base(vm_user, vm_host)
            + ["--exclude", "run_logs/", "--exclude", "summary.json", f"{output_dir}/", f"{vm_user}@{vm_host}:{remote_output}/"]
        )
        summary.finish_stage("sync", True)

        summary.start_stage("remote_run", remote_summary_path=str(remote_summary))
        remote_cmd = " ".join(shlex.quote(part) for part in remote_command_args(args, remote_runner, remote_output, remote_summary))
        rc = run_streamed(ssh_base(vm_user, vm_host) + [remote_cmd], check=False)
        if rc != 0:
            remote_error = f"Remote run.py exited with code {rc}"
            summary.finish_stage("remote_run", False, error=remote_error)
        else:
            summary.finish_stage("remote_run", True)
    except Exception as exc:
        remote_error = str(exc)
        if summary.data["stages"].get("sync", {}).get("success") is None:
            summary.finish_stage("sync", False, error=remote_error)
        elif summary.data["stages"].get("remote_run", {}).get("success") is None:
            summary.finish_stage("remote_run", False, error=remote_error)
    finally:
        summary.finalize(remote_error is None, error=remote_error)
        host_data = json.loads(json.dumps(summary.data))
        fetched = fetch_remote_summary(vm_user, vm_host, remote_summary, local_remote_summary_path)
        host_data["remote_summary_fetched"] = fetched
        host_data["local_remote_summary_path"] = str(local_remote_summary_path)
        if fetched:
            shutil.copy2(local_remote_summary_path, local_summary_path)
        attach_vm_host_summary(local_summary_path, host_data)
        compatibility_summary_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_summary_path, compatibility_summary_path)

    if remote_error:
        raise RuntimeError(remote_error)
    print("SeedEMU run.py VM build/up mode completed successfully.", flush=True)
    return 0


def main():
    args = parse_args()
    output_dir = core.resolve_output_dir(args.output_dir)
    if args.vm:
        summary_path = local_summary_path(output_dir, args.summary_path) if args.summary_path else None
        try:
            return run_on_vm(args, output_dir, summary_path)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    return run_local(args, output_dir, local_summary_path(output_dir, args.summary_path))


if __name__ == "__main__":
    sys.exit(main())
