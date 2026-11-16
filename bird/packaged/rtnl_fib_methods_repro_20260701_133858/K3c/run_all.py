#!/usr/bin/env python3
"""Run a SeedEMU Docker output end-to-end.

Inputs: a SeedEMU Docker compiler output directory containing docker-compose.yml.
Outputs: Docker images/containers, BIRD processes and kernel protocol updates
inside eligible containers, logs under run_logs and logs/dockerBuild, and
summary.json.
Side effects: may build images, start containers, and use nsenter/sudo to start
BIRD or rewrite kernel.conf inside container network namespaces. In --vm mode,
syncs this runner and the output directory to the target VM before executing
remotely.
"""

import argparse
import contextlib
import datetime as dt
import getpass
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BUILD_BATCH = 50
DEFAULT_BUILD_JOBS = 4
DEFAULT_BUILD_MODE = "fast"
DEFAULT_UP_BATCH = 50
DEFAULT_UP_JOBS = 8
DEFAULT_START_BIRD_DELAY = 0.02
DEFAULT_BIRD_START_CMD = "bird"
DEFAULT_VERIFY_BIRD_DELAY = 0.0
DEFAULT_VERIFY_BIRD_JOBS = 16
DEFAULT_PID_CACHE_FILE = "container_pids.json"
DEFAULT_KERNEL_LOAD_THRESHOLD = 50.0
DEFAULT_KERNEL_LOAD_CHECK_INTERVAL = 30
DEFAULT_KERNEL_PER_ROUTER_DELAY = 0.5
DEFAULT_NSENTER_TIMEOUT = 120
DEFAULT_ENTRY_METHOD = "docker-exec"
DEFAULT_START_STAGE_JOBS = 16
DEFAULT_KERNEL_CONFIG_MODE = "build-context"
FAST_DOCKER_BUILDER_SCRIPT = "1buildFastDockerImages.py"
SKIP_BUILD_UP_SERVICES = ("seedemu_internet_map",)
ROUTER_NAME_REGEX = re.compile(r"^as(\d+)brd-r")
KERNEL_CONF_DEST = "/etc/bird/conf/kernel.conf"
KERNEL_EXPORT_ALL_TEMPLATE = """
protocol kernel kernel1 {{
    merge paths on;
    persist;
    scan time {interval};
    ipv4 {{
        import none;
        export all;
    }};
}}
"""
KERNEL_CONFIG_CHECK_CMD = "bird -p -c /etc/bird/bird.conf"
KERNEL_CONFIG_APPLY_CMD = "birdc configure"


class StageError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class SummaryRecorder:
    def __init__(self, path, initial):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stage_started_at = {}
        self._started_at = time.monotonic()
        self.data = {
            "schema_version": 1,
            "started_at": now_iso(),
            "ended_at": None,
            "duration_seconds": None,
            "success": None,
            "error": None,
            "stages": {},
        }
        self.data.update(to_jsonable(initial))
        self.write()

    def write(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(self.data), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        tmp.replace(self.path)

    def update(self, **fields):
        self.data.update(to_jsonable(fields))
        self.write()

    def start_stage(self, name, **fields):
        self._stage_started_at[name] = time.monotonic()
        stage = {
            "success": None,
            "skipped": False,
            "start_time": now_iso(),
            "end_time": None,
            "duration_seconds": None,
            "error": None,
        }
        stage.update(to_jsonable(fields))
        self.data["stages"][name] = stage
        self.write()

    def update_stage(self, name, **fields):
        stage = self.data["stages"].setdefault(name, {})
        stage.update(to_jsonable(fields))
        self.write()

    def finish_stage(self, name, success, error=None, **fields):
        stage = self.data["stages"].setdefault(name, {})
        stage.update(to_jsonable(fields))
        stage["success"] = bool(success)
        stage["error"] = str(error) if error else None
        stage["end_time"] = now_iso()
        started_at = self._stage_started_at.get(name)
        if started_at is not None:
            stage["duration_seconds"] = round(time.monotonic() - started_at, 3)
        self.write()

    def skip_stage(self, name, reason, **fields):
        self.start_stage(name, skipped=True, skip_reason=reason, **fields)
        self.finish_stage(name, True, skipped=True, skip_reason=reason, **fields)

    def finalize(self, success, error=None, **fields):
        self.data.update(to_jsonable(fields))
        self.data["success"] = bool(success)
        self.data["error"] = str(error) if error else None
        self.data["ended_at"] = now_iso()
        self.data["duration_seconds"] = round(time.monotonic() - self._started_at, 3)
        self.write()


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def close(self):
        self._fh.close()

    def log(self, msg):
        line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def log_command(self, cmd, cwd):
        rendered = shlex.join(cmd)
        if len(rendered) > 2000:
            rendered = f"{rendered[:2000]} ... [truncated {len(rendered) - 2000} chars]"
        self.log(f"$ (cd {cwd} && {rendered})")


class TeeWriter:
    """Mirror redirected stdout/stderr into the terminal and a stage log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync/build/up a SeedEMU Docker output, start BIRD, and update kernel export."
    )
    parser.add_argument(
        "output_dir",
        help="Path to a SeedEMU compiler output directory containing docker-compose.yml.",
    )
    parser.add_argument("--project-name", help="Docker Compose project name.")
    parser.add_argument(
        "--build-mode",
        choices=["fast", "compose"],
        default=DEFAULT_BUILD_MODE,
        help="Build images with fast archive generation or the legacy docker compose batch flow.",
    )
    parser.add_argument("--build-batch", type=int, default=DEFAULT_BUILD_BATCH)
    parser.add_argument("--build-jobs", type=int, default=DEFAULT_BUILD_JOBS)
    parser.add_argument("--up-batch", type=int, default=DEFAULT_UP_BATCH)
    parser.add_argument("--up-jobs", type=int, default=DEFAULT_UP_JOBS)
    parser.add_argument(
        "--down-first",
        action="store_true",
        help="Run docker compose down before build/up for this project.",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip build.")
    parser.add_argument("--no-up", action="store_true", help="Skip up.")
    parser.add_argument(
        "--no-start-bird",
        action="store_true",
        help="Skip BIRD startup after docker compose up.",
    )
    parser.add_argument(
        "--no-start-kernel",
        action="store_true",
        help="Skip kernel.conf export-all update after BIRD startup.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Accepted for compatibility; ignored by the buildTest1-compatible flow.",
    )
    parser.add_argument(
        "--summary-path",
        help="Path for summary.json. Default: <output_dir>/summary.json.",
    )
    parser.add_argument(
        "--pid-cache-file",
        default=DEFAULT_PID_CACHE_FILE,
        help="PID cache file written by start-bird. Relative paths use output_dir parent.",
    )
    parser.add_argument(
        "--container-name-filter",
        help="Optional docker ps name filter in addition to the Compose project label.",
    )
    parser.add_argument(
        "--bird-start-command",
        default=DEFAULT_BIRD_START_CMD,
        help="Shell command used inside each eligible container to start BIRD.",
    )
    parser.add_argument(
        "--start-bird-delay",
        type=float,
        default=DEFAULT_START_BIRD_DELAY,
        help="Delay between BIRD start triggers, in seconds.",
    )
    parser.add_argument(
        "--verify-bird",
        action="store_true",
        help="Run a separate verification stage after BIRD start triggers.",
    )
    parser.add_argument(
        "--verify-bird-delay",
        type=float,
        default=DEFAULT_VERIFY_BIRD_DELAY,
        help="Seconds to wait before the optional verify-bird stage.",
    )
    parser.add_argument(
        "--verify-bird-jobs",
        type=int,
        default=DEFAULT_VERIFY_BIRD_JOBS,
        help="Parallel nsenter checks for the optional verify-bird stage.",
    )
    parser.add_argument(
        "--kernel-load-threshold",
        type=float,
        default=DEFAULT_KERNEL_LOAD_THRESHOLD,
        help="Wait until 1-minute load average is below this value between kernel updates.",
    )
    parser.add_argument(
        "--kernel-load-check-interval",
        type=float,
        default=DEFAULT_KERNEL_LOAD_CHECK_INTERVAL,
        help="Seconds between load checks during start-kernel.",
    )
    parser.add_argument(
        "--kernel-per-router-delay",
        type=float,
        default=DEFAULT_KERNEL_PER_ROUTER_DELAY,
        help="Delay after each router kernel.conf update, in seconds.",
    )
    parser.add_argument(
        "--kernel-config-mode",
        choices=["build-context", "runtime"],
        default=DEFAULT_KERNEL_CONFIG_MODE,
        help=(
            "build-context rewrites SeedEMU output kernel.conf before build and skips runtime birdc configure; "
            "runtime keeps the legacy post-start per-container update."
        ),
    )
    parser.add_argument(
        "--entry-method",
        choices=["docker-exec", "nsenter"],
        default=DEFAULT_ENTRY_METHOD,
        help="Container command entry method used by start-bird/start-kernel runtime operations.",
    )
    parser.add_argument(
        "--start-stage-jobs",
        type=int,
        default=DEFAULT_START_STAGE_JOBS,
        help="Parallel command entries for optimized docker-exec start stages.",
    )
    parser.add_argument(
        "--nsenter-sudo",
        choices=["auto", "always", "never"],
        default="auto",
        help="Use sudo -n for nsenter when not running as root.",
    )
    parser.add_argument(
        "--nsenter-timeout",
        type=float,
        default=DEFAULT_NSENTER_TIMEOUT,
        help="Timeout in seconds for synchronous nsenter commands.",
    )
    parser.add_argument(
        "--vm",
        help="Sync output_dir to this VM over SSH, then run this script on the VM.",
    )
    parser.add_argument(
        "--vm-user",
        default="lxl",
        help="SSH user for --vm mode. Default: lxl.",
    )
    parser.add_argument(
        "--vm-output-root",
        help=(
            "Remote VM run root. Default: /home/<vm-user>/bird/runs. "
            "Each --vm run creates topology-<service-count>_<timestamp>_<project>/ under it."
        ),
    )
    parser.add_argument(
        "--vm-runner",
        help="Remote path for this script. Default: /home/<vm-user>/bird/run_all.py.",
    )
    return parser.parse_args()


def resolve_output_dir(raw):
    path = Path(raw).expanduser().resolve()
    if (path / "docker-compose.yml").is_file():
        return path
    nested = path / "output"
    if (nested / "docker-compose.yml").is_file():
        return nested.resolve()
    raise SystemExit(f"Cannot find docker-compose.yml in {path} or {nested}")


def resolve_summary_path(raw, output_dir):
    if raw:
        return Path(raw).expanduser().resolve()
    return output_dir / "summary.json"


def resolve_pid_cache_file(raw, cwd):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def default_project_name(output_dir):
    base = output_dir.parent.name if output_dir.name == "output" else output_dir.name
    slug = re.sub(r"[^a-z0-9_-]+", "-", base.lower()).strip("-")
    if not slug:
        slug = "seedemu"
    digest = hashlib.sha1(str(output_dir).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def slugify(value, fallback="seedemu"):
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-")
    return slug or fallback


def compose_service_names_from_file(output_dir):
    compose_file = output_dir / "docker-compose.yml"
    names = []
    in_services = False
    services_indent = 0
    item_indent = None
    for raw_line in compose_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = raw_line.strip()
        if not in_services:
            if indent == 0 and stripped == "services:":
                in_services = True
                services_indent = indent
            continue
        if indent <= services_indent:
            break
        match = re.match(r"([A-Za-z0-9_.-]+):(?:\s*(?:#.*)?)?$", stripped)
        if not match:
            continue
        if item_indent is None:
            item_indent = indent
        if indent == item_indent:
            names.append(match.group(1))
    return names


def remote_run_paths(output_dir, project_name, vm_user, vm_output_root, timestamp=None):
    root = Path(vm_output_root or f"/home/{vm_user}/bird/runs")
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    services = compose_service_names_from_file(output_dir)
    service_count = len(services) if services else "unknown"
    project_slug = slugify(project_name, "project")
    run_dir = root / f"topology-{service_count}_{timestamp}_{project_slug}"
    return run_dir, run_dir / "output", service_count


def local_vm_summary_path(topology_size, timestamp):
    run_dir = Path(__file__).resolve().parent / "runs" / f"topology-{topology_size}_{timestamp}"
    return run_dir / "summary.json"


def compose_cmd(output_dir, project_name):
    cmd = ["docker", "compose", "-f", str(output_dir / "docker-compose.yml")]
    if project_name:
        cmd += ["-p", project_name]
    return cmd


def run_logged(cmd, cwd, logger, env=None):
    logger.log_command(cmd, cwd)
    with logger.path.open("a", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")


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


def command_output(cmd, cwd, logger):
    logger.log_command(cmd, cwd)
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.stderr:
        logger._fh.write(proc.stderr)
        logger._fh.flush()
    if proc.returncode != 0:
        if proc.stdout:
            logger._fh.write(proc.stdout)
            logger._fh.flush()
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")
    return proc.stdout


def get_services(base_cmd, cwd, logger):
    raw = command_output(base_cmd + ["config", "--services"], cwd, logger)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def filter_build_up_services(services, logger=None):
    skipped = [name for name in services if name in SKIP_BUILD_UP_SERVICES]
    active = [name for name in services if name not in SKIP_BUILD_UP_SERVICES]
    if skipped and logger:
        logger.log(
            "Skipping build/up service(s): "
            + ", ".join(skipped)
            + " (fixed container_name can conflict across SeedEMU runs)"
        )
    return active, skipped


def chunks(items, size):
    if size <= 0:
        raise ValueError("batch size must be > 0")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def script_log_dir(cwd):
    base = Path(os.environ.get("EXP_LOG_DIR", "./logs"))
    if not base.is_absolute():
        base = cwd / base
    path = base / "dockerBuild"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_log_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_file(src, dst):
    if not src.exists():
        return
    with src.open("rb") as in_fh, dst.open("ab") as out_fh:
        while True:
            chunk = in_fh.read(1024 * 1024)
            if not chunk:
                break
            out_fh.write(chunk)


def fast_builder_candidates():
    here = Path(__file__).resolve()
    return [
        here.with_name(FAST_DOCKER_BUILDER_SCRIPT),
        here.parent.parent / FAST_DOCKER_BUILDER_SCRIPT,
    ]


def resolve_fast_builder_script():
    for path in fast_builder_candidates():
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in fast_builder_candidates())
    raise RuntimeError(f"Cannot find {FAST_DOCKER_BUILDER_SCRIPT}; searched: {searched}")


def load_fast_builder_module():
    script_path = resolve_fast_builder_script()
    spec = importlib.util.spec_from_file_location("seedemu_fast_docker_builder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import fast Docker builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, script_path


def run_fastdocker_flow(base_cmd, output_dir, cwd, logger, project_name, summary):
    started = time.monotonic()
    log_file = script_log_dir(cwd) / f"dockerBuild_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    write_log_header(log_file)

    compose_file = output_dir / "docker-compose.yml"
    work_dir = output_dir / ".fastDockerBuild"
    archive_file = work_dir / "images_all.tar"
    fast_summary_file = work_dir / "summary.json"
    build_images_file = work_dir / "build_images.sh"

    info = {
        "mode": "fast_archive",
        "log_file": str(log_file),
        "compose_file": str(compose_file),
        "work_dir": str(work_dir),
        "archive_file": str(archive_file),
        "fast_summary_file": str(fast_summary_file),
        "build_images_file": str(build_images_file),
        "project_name": project_name,
        "services_discovered": 0,
        "services_total": 0,
        "services_skipped": 0,
        "skipped_services": [],
        "image_count": 0,
        "normal_image_count": 0,
        "base_count": 0,
        "archive_bytes": 0,
    }
    summary.update_stage("build", **info)

    services = get_services(base_cmd, cwd, logger)
    if not services:
        raise StageError("docker compose config --services returned no services", info)
    active_services, skipped_services = filter_build_up_services(services, logger)
    if not active_services:
        raise StageError("no services left after applying build/up skip list", info)

    info.update(
        {
            "services_discovered": len(services),
            "services_total": len(active_services),
            "services_skipped": len(skipped_services),
            "skipped_services": skipped_services,
        }
    )
    summary.update_stage("build", **info)

    logger.log("Checking build context size (target: output directory)...")
    fast_builder, helper_path = load_fast_builder_module()
    info["helper_script"] = str(helper_path)
    summary.update_stage("build", **info)
    logger.log(f"Using fast Docker image archive builder: {helper_path}")

    saved_env = {
        "DOCKER_BUILDKIT": os.environ.get("DOCKER_BUILDKIT"),
        "COMPOSE_DOCKER_CLI_BUILD": os.environ.get("COMPOSE_DOCKER_CLI_BUILD"),
        "COMPOSE_PROJECT_NAME": os.environ.get("COMPOSE_PROJECT_NAME"),
    }

    try:
        os.environ["DOCKER_BUILDKIT"] = "1"
        os.environ["COMPOSE_DOCKER_CLI_BUILD"] = "1"
        os.environ["COMPOSE_PROJECT_NAME"] = project_name

        with log_file.open("a", encoding="utf-8") as build_log:
            tee_out = TeeWriter(sys.stdout, build_log)
            tee_err = TeeWriter(sys.stderr, build_log)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                fast_builder.requireCommand("docker")
                fast_builder.runCommand(["du", "-sh", str(output_dir)])
                fast_summary = fast_builder.createFastArchive(
                    compose_file=compose_file,
                    archive_file=archive_file,
                    summary_file=fast_summary_file,
                    build_images_file=build_images_file,
                    cache_dir=work_dir / "cache",
                    skip_services=skipped_services,
                )
                logger.log(
                    "Fast archive generated: "
                    f"images={fast_summary['imageCount']} "
                    f"normal={fast_summary['normalImageCount']} "
                    f"bases={fast_summary['baseCount']} "
                    f"archive={archive_file}"
                )
                fast_builder.runCommand(["docker", "load", "-i", str(archive_file)])
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    duration = int(time.monotonic() - started)
    fast_summary = json.loads(fast_summary_file.read_text(encoding="utf-8"))
    info.update(
        {
            "duration_seconds": duration,
            "image_count": fast_summary.get("imageCount", 0),
            "normal_image_count": fast_summary.get("normalImageCount", 0),
            "base_count": fast_summary.get("baseCount", 0),
            "archive_bytes": fast_summary.get("archiveBytes", 0),
            "fast_duration_seconds": fast_summary.get("durationSeconds"),
        }
    )

    minutes = (duration % 3600) // 60
    seconds = duration % 60
    runtime_msg = f"Total runtime: {minutes}m {seconds}s"
    logger.log(runtime_msg)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(runtime_msg + "\n")
        fh.write(f"Full log saved to: {log_file}\n")
    logger.log(f"Full log saved to: {log_file}")
    return info


def run_buildtest1_flow(base_cmd, output_dir, cwd, logger, batch_size, parallel_jobs, summary):
    started = time.monotonic()
    log_file = script_log_dir(cwd) / f"dockerBuild_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    write_log_header(log_file)

    info = {
        "log_file": str(log_file),
        "batch_size": batch_size,
        "parallel_jobs": parallel_jobs,
        "services_total": 0,
        "batches_total": 0,
        "batches_succeeded": 0,
        "batches_failed": 0,
        "failed_batches": [],
    }
    summary.update_stage("build", **info)

    logger.log("Checking build context size (target: output directory)...")
    run_logged(["du", "-sh", str(output_dir)], cwd, logger)

    logger.log("-------------------------------------------------")
    logger.log("Phase 1: Cache Warmer (Standard)")
    logger.log("-------------------------------------------------")
    services = get_services(base_cmd, cwd, logger)
    if not services:
        raise StageError("docker compose config --services returned no services", info)
    all_services = services
    services, skipped_services = filter_build_up_services(all_services, logger)
    if not services:
        raise StageError("no services left after applying build/up skip list", info)

    first_service = services[0]
    warmup_log = cwd / "warmup.log"
    info.update(
        {
            "services_discovered": len(all_services),
            "services_total": len(services),
            "services_skipped": len(skipped_services),
            "skipped_services": skipped_services,
            "warmup_service": first_service,
            "warmup_success": None,
        }
    )
    summary.update_stage("build", **info)
    logger.log(f"Building first service: {first_service} ...")

    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"
    env["COMPOSE_DOCKER_CLI_BUILD"] = "1"

    with warmup_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            base_cmd + ["build", first_service],
            cwd=cwd,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        logger.log("Warmup failed. Error log:")
        logger._fh.write(warmup_log.read_text(encoding="utf-8", errors="replace"))
        logger._fh.flush()
        warmup_log.unlink(missing_ok=True)
        info["warmup_success"] = False
        raise StageError(f"Warmup build failed for {first_service}", info)

    append_file(warmup_log, log_file)
    warmup_log.unlink(missing_ok=True)
    info["warmup_success"] = True
    logger.log("Warmup successful.")
    summary.update_stage("build", **info)

    logger.log("-------------------------------------------------")
    logger.log(f"Phase 2: Golden Ratio Build (P={parallel_jobs}, B={batch_size})")
    logger.log("-------------------------------------------------")

    batches = list(chunks(services, batch_size))
    info["batches_total"] = len(batches)
    batch_list = cwd / "batch_list.txt"
    batch_list.write_text("\n".join(" ".join(batch) for batch in batches) + "\n", encoding="utf-8")
    logger.log(f"Total Batches to run: {len(batches)}")
    summary.update_stage("build", **info)

    def build_batch(batch):
        batch_started = time.monotonic()
        batch_text = " ".join(batch)
        batch_id = hashlib.md5(batch_text.encode("utf-8")).hexdigest()[:5]
        temp_log = cwd / f"build_log_{batch_id}.txt"
        with temp_log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                base_cmd + ["build"] + batch,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return {
            "batch_id": batch_id,
            "returncode": proc.returncode,
            "temp_log": temp_log,
            "duration_seconds": int(time.monotonic() - batch_started),
        }

    temp_logs = []
    with ThreadPoolExecutor(max_workers=max(1, parallel_jobs)) as pool:
        futures = [pool.submit(build_batch, batch) for batch in batches]
        for future in as_completed(futures):
            result = future.result()
            temp_logs.append(result["temp_log"])
            if result["returncode"] == 0:
                info["batches_succeeded"] += 1
                logger.log(f"Batch {result['batch_id']} done in {result['duration_seconds']}s")
            else:
                info["batches_failed"] += 1
                info["failed_batches"].append(result["batch_id"])
                logger.log(
                    f"Batch {result['batch_id']} FAILED in "
                    f"{result['duration_seconds']}s (Check {result['temp_log'].name})"
                )
            summary.update_stage("build", **info)

    logger.log("-------------------------------------------------")
    logger.log(f"Aggregating logs to {log_file} ...")
    for temp_log in temp_logs:
        append_file(temp_log, log_file)

    batch_list.unlink(missing_ok=True)
    for temp_log in temp_logs:
        temp_log.unlink(missing_ok=True)

    duration = int(time.monotonic() - started)
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    runtime_msg = f"Total runtime: {minutes}m {seconds}s"
    logger.log(runtime_msg)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(runtime_msg + "\n")
        fh.write(f"Full log saved to: {log_file}\n")
    logger.log(f"Full log saved to: {log_file}")

    info["duration_seconds"] = duration
    if info["batches_failed"]:
        failed = ", ".join(info["failed_batches"])
        raise StageError(f"Build failed for batches: {failed}", info)
    return info


def run_dockerup_flow(base_cmd, cwd, logger, batch_size, parallel_jobs, summary):
    started = time.monotonic()
    log_file = script_log_dir(cwd) / f"dockerUp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    info = {
        "log_file": str(log_file),
        "batch_size": batch_size,
        "parallel_jobs": parallel_jobs,
        "services_total": 0,
        "batches_total": 0,
        "batches_succeeded": 0,
        "batches_failed": 0,
        "failed_batches": [],
    }
    summary.update_stage("up", **info)

    def log_output(msg):
        logger.log(msg)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

    log_output("===== Docker Compose Up batch startup started =====")
    log_output(f"Start time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_output(f"Compose file: {base_cmd[3]}")
    log_output(f"Parallel jobs: {parallel_jobs}")
    log_output(f"Batch size: {batch_size}")
    log_output("Fetching service names from compose file...")

    services = get_services(base_cmd, cwd, logger)
    if not services:
        raise StageError("docker compose config --services returned no services", info)
    all_services = services
    services, skipped_services = filter_build_up_services(all_services, logger)
    if not services:
        raise StageError("no services left after applying build/up skip list", info)

    batches = list(chunks(services, batch_size))
    info.update(
        {
            "services_discovered": len(all_services),
            "services_total": len(services),
            "services_skipped": len(skipped_services),
            "skipped_services": skipped_services,
            "batches_total": len(batches),
        }
    )
    summary.update_stage("up", **info)

    log_output("Starting parallel docker compose up -d...")
    log_output(f"  Parallel Jobs (-P): {parallel_jobs}")
    log_output(f"  Batch Size (-n):    {batch_size}")

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

    log_output("-------------------------------------")
    log_output("All up batches finished.")

    duration = int(time.monotonic() - started)
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    log_output("-------------------------------------")
    log_output(f"Total runtime: {hours}h {minutes}m {seconds}s")
    log_output("===== Docker Compose Up batch startup ended =====")
    log_output(f"Full log path: {log_file}")

    info["duration_seconds"] = duration
    if info["batches_failed"]:
        raise StageError(f"docker compose up failed for {info['batches_failed']} batch(es)", info)
    return info


def docker_output(cmd, cwd, logger=None):
    if logger:
        logger.log_command(cmd, cwd)
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or shlex.join(cmd)
        raise RuntimeError(msg)
    return proc.stdout


def project_container_ids(project_name, cwd, logger, name_filter=None):
    cmd = [
        "docker",
        "ps",
        "--filter",
        f"label=com.docker.compose.project={project_name}",
    ]
    if name_filter:
        cmd += ["--filter", f"name={name_filter}"]
    cmd += ["--format", "{{.ID}}"]
    raw = docker_output(cmd, cwd, logger)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def inspect_containers(container_ids, cwd, logger):
    if not container_ids:
        return []
    cmd = [
        "docker",
        "inspect",
        "-f",
        "{{.Id}} {{.Name}} {{.State.Pid}} {{.State.Running}}",
    ] + container_ids
    raw = docker_output(cmd, cwd, logger)
    containers = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cid = parts[0]
        name = parts[1].lstrip("/")
        try:
            pid = int(parts[2])
        except ValueError:
            pid = 0
        containers.append(
            {
                "id": cid,
                "short_id": cid[:12],
                "name": name,
                "pid": pid,
                "running": parts[3].lower() == "true",
            }
        )
    return containers


def nsenter_prefix(mode):
    if os.geteuid() == 0:
        return ["nsenter"]
    if mode == "never":
        raise RuntimeError("nsenter stages require root when --nsenter-sudo=never")
    if shutil.which("sudo"):
        check = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            return ["sudo", "-n", "nsenter"]
    if mode == "always":
        raise RuntimeError("sudo -n is required for nsenter but is not available")
    raise RuntimeError("nsenter stages require root or passwordless sudo")


def timeout_completed_process(cmd, exc):
    stdout = exc.stdout or ""
    stderr = exc.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    message = f"timeout after {exc.timeout}s"
    stderr = f"{stderr}\n{message}".strip() if stderr else message
    return subprocess.CompletedProcess(cmd, 124, stdout=stdout, stderr=stderr)


def run_nsenter_cmd(prefix, pid, shell_cmd, async_mode=False, timeout=None):
    cmd = prefix + ["-t", str(pid), "-n", "-m", "--", "sh", "-c", shell_cmd]
    if async_mode:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return timeout_completed_process(cmd, exc)


def write_pid_cache(pid_map, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(pid_map, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_nsenter_file(prefix, pid, remote_path, content, timeout=None):
    cmd = prefix + ["-t", str(pid), "-n", "-m", "--", "tee", remote_path]
    try:
        return subprocess.run(cmd, input=content, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return timeout_completed_process(cmd, exc)


def run_docker_exec_cmd(container, shell_cmd, async_mode=False, timeout=None, input_text=None):
    cmd = ["docker", "exec"]
    if async_mode:
        cmd.append("-d")
    elif input_text is not None:
        cmd.append("-i")
    cmd += [container["short_id"], "sh", "-c", shell_cmd]
    try:
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return timeout_completed_process(cmd, exc)


def run_container_cmd(method, prefix, container, shell_cmd, async_mode=False, timeout=None, input_text=None):
    if method == "docker-exec":
        return run_docker_exec_cmd(container, shell_cmd, async_mode=async_mode, timeout=timeout, input_text=input_text)
    if input_text is not None:
        cmd = prefix + ["-t", str(container["pid"]), "-n", "-m", "--", "sh", "-c", shell_cmd]
        try:
            return subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return timeout_completed_process(cmd, exc)
    return run_nsenter_cmd(prefix, container["pid"], shell_cmd, async_mode=async_mode, timeout=timeout)


def stable_kernel_interval(name):
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return 6000 + (int(digest[:8], 16) % 1201)


def kernel_conf_sources(output_dir):
    sources = []
    for dockerfile in sorted(output_dir.glob("*/Dockerfile")):
        context_dir = dockerfile.parent
        try:
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                fields = shlex.split(stripped)
            except ValueError:
                continue
            if len(fields) == 3 and fields[0].upper() == "COPY" and fields[2] == KERNEL_CONF_DEST:
                source = (context_dir / fields[1]).resolve()
                sources.append((context_dir.name, source))
                break
    return sources


def prepare_kernel_context_flow(output_dir, logger, args, summary):
    sources = kernel_conf_sources(output_dir)
    info = {
        "mode": "build-context",
        "output_dir": str(output_dir),
        "kernel_conf_dest": KERNEL_CONF_DEST,
        "files_total": len(sources),
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "failed_files": [],
    }
    summary.update_stage("prepare_kernel", **info)
    logger.log("Preparing kernel.conf export-all in SeedEMU build contexts...")
    for context_name, source in sources:
        interval = stable_kernel_interval(context_name)
        content = KERNEL_EXPORT_ALL_TEMPLATE.format(interval=interval)
        try:
            old = source.read_text(encoding="utf-8") if source.exists() else None
            if old == content:
                info["unchanged"] += 1
            else:
                source.write_text(content, encoding="utf-8")
                info["updated"] += 1
        except OSError as exc:
            info["failed"] += 1
            info["failed_files"].append({"path": str(source), "error": str(exc)})
        if (info["updated"] + info["unchanged"] + info["failed"]) % 200 == 0:
            logger.log(
                "prepare-kernel progress: "
                f"{info['updated'] + info['unchanged'] + info['failed']}/{info['files_total']} "
                f"updated={info['updated']} unchanged={info['unchanged']} failed={info['failed']}"
            )
        summary.update_stage("prepare_kernel", **info)
    logger.log(
        "prepare-kernel finished: "
        f"files={info['files_total']} updated={info['updated']} "
        f"unchanged={info['unchanged']} failed={info['failed']}"
    )
    if info["failed"]:
        raise StageError(f"prepare-kernel failed for {info['failed']} file(s)", info)
    if info["files_total"] == 0:
        raise StageError("No kernel.conf build-context sources found", info)
    return info


def start_bird_flow(project_name, cwd, logger, args, pid_cache_file, summary):
    info = {
        "project_name": project_name,
        "pid_cache_file": str(pid_cache_file),
        "container_name_filter": args.container_name_filter,
        "containers_total": 0,
        "containers_running": 0,
        "eligible_containers": 0,
        "triggered": 0,
        "started": 0,
        "already_running": 0,
        "skipped_no_bird": 0,
        "skipped_not_running": 0,
        "failed": 0,
        "failed_containers": [],
        "assume_started_after_trigger": True,
        "verification": "deferred_to_verify_bird_stage",
        "entry_method": args.entry_method,
        "parallel_jobs": args.start_stage_jobs if args.entry_method == "docker-exec" else 1,
        "nsenter_timeout_seconds": args.nsenter_timeout,
    }
    summary.update_stage("start_bird", **info)
    logger.log("Scanning project containers for BIRD startup...")

    prefix = nsenter_prefix(args.nsenter_sudo) if args.entry_method == "nsenter" else None
    container_ids = project_container_ids(project_name, cwd, logger, args.container_name_filter)
    containers = inspect_containers(container_ids, cwd, logger)
    pid_map = {container["name"]: container["pid"] for container in containers if container["pid"] > 0}
    write_pid_cache(pid_map, pid_cache_file)

    info["containers_total"] = len(containers)
    info["containers_running"] = sum(1 for container in containers if container["running"])
    summary.update_stage("start_bird", **info)

    logger.log(f"Found {len(containers)} project containers; {info['containers_running']} running.")

    if args.entry_method == "docker-exec":
        running_containers = [container for container in containers if container["running"] and container["pid"] > 0]
        info["eligible_containers"] = len(running_containers)
        summary.update_stage("start_bird", **info)
        start_cmd = (
            "command -v bird >/dev/null 2>&1 "
            "&& test -f /etc/bird/bird.conf "
            f"&& exec {args.bird_start_command}"
        )

        def trigger(container):
            result = run_docker_exec_cmd(container, start_cmd, async_mode=True, timeout=args.nsenter_timeout)
            error = ""
            if result.returncode != 0:
                error = (result.stderr or result.stdout or "").strip()
            return container["name"], result.returncode, error

        count = 0
        with ThreadPoolExecutor(max_workers=max(1, args.start_stage_jobs)) as pool:
            futures = []
            for container in running_containers:
                futures.append(pool.submit(trigger, container))
                if args.start_bird_delay > 0:
                    time.sleep(args.start_bird_delay)
            for future in as_completed(futures):
                name, returncode, error = future.result()
                info["triggered"] += 1
                if returncode == 0:
                    info["started"] += 1
                else:
                    info["failed"] += 1
                    info["failed_containers"].append({"name": name, "stage": "docker_exec_start", "error": error})
                count += 1
                if count % 200 == 0:
                    logger.log(
                        f"Started BIRD trigger in {count}/{info['eligible_containers']} containers "
                        f"with docker exec..."
                    )
                summary.update_stage("start_bird", **info)

        logger.log(
            "BIRD trigger finished: "
            f"eligible={info['eligible_containers']} triggered={info['triggered']} "
            f"failed={info['failed']} entry_method=docker-exec "
            "verification=deferred_to_verify_bird_stage"
        )
        if info["eligible_containers"] == 0:
            raise StageError("No eligible BIRD containers found", info)
        if info["failed"] > 0:
            raise StageError(f"Failed to trigger BIRD in {info['failed']} containers", info)
        return info

    count = 0
    for container in containers:
        name = container["name"]
        pid = container["pid"]
        if not container["running"] or pid <= 0:
            info["skipped_not_running"] += 1
            summary.update_stage("start_bird", **info)
            continue

        try:
            probe = run_nsenter_cmd(
                prefix,
                pid,
                "command -v bird >/dev/null 2>&1 && test -f /etc/bird/bird.conf",
                timeout=args.nsenter_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            info["failed"] += 1
            info["failed_containers"].append(
                {"name": name, "stage": "probe", "error": f"timeout after {exc.timeout}s"}
            )
            summary.update_stage("start_bird", **info)
            continue
        if probe.returncode == 124:
            info["failed"] += 1
            info["failed_containers"].append(
                {"name": name, "stage": "probe", "error": (probe.stderr or probe.stdout).strip()}
            )
            summary.update_stage("start_bird", **info)
            continue
        if probe.returncode != 0:
            info["skipped_no_bird"] += 1
            summary.update_stage("start_bird", **info)
            continue

        info["eligible_containers"] += 1
        try:
            run_nsenter_cmd(prefix, pid, args.bird_start_command, async_mode=True)
            info["triggered"] += 1
            info["started"] += 1
            count += 1
            if count % 200 == 0:
                logger.log(f"Started BIRD trigger in {count} containers...")
                time.sleep(5)
            if args.start_bird_delay > 0:
                time.sleep(args.start_bird_delay)
        except Exception as exc:
            info["failed"] += 1
            info["failed_containers"].append({"name": name, "error": str(exc)})
        summary.update_stage("start_bird", **info)

    logger.log(
        "BIRD trigger finished: "
        f"eligible={info['eligible_containers']} triggered={info['triggered']} "
        f"failed={info['failed']} skipped_no_bird={info['skipped_no_bird']} "
        "verification=deferred_to_verify_bird_stage"
    )

    if info["eligible_containers"] == 0:
        raise StageError("No eligible BIRD containers found", info)
    if info["failed"] > 0:
        raise StageError(f"Failed to trigger BIRD in {info['failed']} containers", info)
    return info


def verify_bird_flow(project_name, cwd, logger, args, summary):
    info = {
        "project_name": project_name,
        "container_name_filter": args.container_name_filter,
        "delay_seconds": args.verify_bird_delay,
        "parallel_jobs": args.verify_bird_jobs,
        "nsenter_timeout_seconds": args.nsenter_timeout,
        "containers_total": 0,
        "containers_running": 0,
        "checked": 0,
        "bird_running": 0,
        "missing_bird": 0,
        "skipped_not_running": 0,
        "failed": 0,
        "failed_containers": [],
    }
    summary.update_stage("verify_bird", **info)

    if args.verify_bird_delay > 0:
        logger.log(f"verify-bird waiting {args.verify_bird_delay}s before checking BIRD processes...")
        time.sleep(args.verify_bird_delay)

    logger.log("Verifying BIRD processes in project containers...")
    prefix = nsenter_prefix(args.nsenter_sudo)
    container_ids = project_container_ids(project_name, cwd, logger, args.container_name_filter)
    containers = inspect_containers(container_ids, cwd, logger)
    running_containers = [container for container in containers if container["running"] and container["pid"] > 0]

    info["containers_total"] = len(containers)
    info["containers_running"] = len(running_containers)
    info["skipped_not_running"] = len(containers) - len(running_containers)
    summary.update_stage("verify_bird", **info)

    def check_container(container):
        proc = run_nsenter_cmd(prefix, container["pid"], "pgrep -x bird >/dev/null", timeout=args.nsenter_timeout)
        return container["name"], proc.returncode, (proc.stderr or proc.stdout or "").strip()

    max_workers = max(1, args.verify_bird_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(check_container, container) for container in running_containers]
        for future in as_completed(futures):
            name, returncode, error = future.result()
            info["checked"] += 1
            if returncode == 0:
                info["bird_running"] += 1
            elif returncode == 1:
                info["missing_bird"] += 1
            else:
                info["failed"] += 1
                info["failed_containers"].append({"name": name, "stage": "verify", "error": error})
            if info["checked"] % 200 == 0:
                logger.log(
                    f"verify-bird checked {info['checked']}/{info['containers_running']}: "
                    f"running={info['bird_running']} missing={info['missing_bird']} failed={info['failed']}"
                )
            summary.update_stage("verify_bird", **info)

    logger.log(
        "verify-bird finished: "
        f"checked={info['checked']} running={info['bird_running']} "
        f"missing={info['missing_bird']} failed={info['failed']}"
    )
    if info["failed"] > 0:
        raise StageError(f"verify-bird failed for {info['failed']} containers", info)
    if info["missing_bird"] > 0:
        raise StageError(f"BIRD is not running in {info['missing_bird']} checked containers", info)
    return info


def group_router_containers(containers):
    groups = {}
    ignored = 0
    for container in containers:
        match = ROUTER_NAME_REGEX.match(container["name"])
        if not match:
            ignored += 1
            continue
        asn = int(match.group(1))
        groups.setdefault(asn, []).append(container)
    return groups, ignored


def wait_for_system_idle(logger, phase_name, threshold, interval):
    logger.log(f"{phase_name}: waiting for 1-minute load average < {threshold}")
    while True:
        try:
            load1, _, _ = os.getloadavg()
        except OSError:
            logger.log("Cannot read system load average; continuing.")
            return None
        if load1 < threshold:
            logger.log(f"{phase_name}: load stabilized at {load1:.2f}")
            return load1
        logger.log(f"{phase_name}: load {load1:.2f} >= {threshold}; sleeping {interval}s")
        time.sleep(max(0.0, interval))


def start_kernel_flow(project_name, cwd, logger, args, summary):
    info = {
        "project_name": project_name,
        "container_name_filter": args.container_name_filter,
        "router_name_regex": ROUTER_NAME_REGEX.pattern,
        "load_threshold": args.kernel_load_threshold,
        "load_check_interval": args.kernel_load_check_interval,
        "per_router_delay": args.kernel_per_router_delay,
        "config_check_cmd": KERNEL_CONFIG_CHECK_CMD,
        "config_apply_cmd": KERNEL_CONFIG_APPLY_CMD,
        "entry_method": args.entry_method,
        "nsenter_timeout_seconds": args.nsenter_timeout,
        "containers_total": 0,
        "containers_running": 0,
        "router_containers": 0,
        "ignored_containers": 0,
        "processed": 0,
        "failed": 0,
        "failed_containers": [],
        "skipped_not_running": 0,
        "skipped_no_kernel_conf": 0,
        "as_groups": {},
    }
    summary.update_stage("start_kernel", **info)
    logger.log("Scanning project router containers for kernel.conf export-all update...")

    prefix = nsenter_prefix(args.nsenter_sudo) if args.entry_method == "nsenter" else None
    container_ids = project_container_ids(project_name, cwd, logger, args.container_name_filter)
    containers = inspect_containers(container_ids, cwd, logger)
    groups, ignored = group_router_containers(containers)

    info["containers_total"] = len(containers)
    info["containers_running"] = sum(1 for container in containers if container["running"])
    info["router_containers"] = sum(len(items) for items in groups.values())
    info["ignored_containers"] = ignored
    info["as_groups"] = {asn: len(items) for asn, items in sorted(groups.items())}
    summary.update_stage("start_kernel", **info)

    if info["router_containers"] == 0:
        raise StageError("No router containers matched for start-kernel", info)

    logger.log(
        f"Selected {info['router_containers']} router containers; "
        f"ignored {info['ignored_containers']} non-router containers."
    )

    for asn in sorted(groups):
        logger.log(f"Processing AS {asn} for kernel.conf update...")
        for container in groups[asn]:
            name = container["name"]
            pid = container["pid"]
            if not container["running"] or pid <= 0:
                info["skipped_not_running"] += 1
                summary.update_stage("start_kernel", **info)
                continue

            interval = 6000 + random.randint(0, 1200)
            new_conf = KERNEL_EXPORT_ALL_TEMPLATE.format(interval=interval)
            combined_cmd = (
                f"test -f {shlex.quote(KERNEL_CONF_DEST)} || exit 42; "
                f"cat > {shlex.quote(KERNEL_CONF_DEST)} && "
                f"{KERNEL_CONFIG_CHECK_CMD} && "
                f"{KERNEL_CONFIG_APPLY_CMD}"
            )
            combined_result = run_container_cmd(
                args.entry_method,
                prefix,
                container,
                combined_cmd,
                input_text=new_conf,
                timeout=args.nsenter_timeout,
            )
            if combined_result.returncode == 124:
                info["failed"] += 1
                info["failed_containers"].append(
                    {"name": name, "stage": "combined_update", "error": (combined_result.stderr or combined_result.stdout).strip()}
                )
                summary.update_stage("start_kernel", **info)
                continue
            if combined_result.returncode == 42:
                info["skipped_no_kernel_conf"] += 1
                summary.update_stage("start_kernel", **info)
                continue
            if combined_result.returncode != 0:
                info["failed"] += 1
                info["failed_containers"].append(
                    {
                        "name": name,
                        "stage": "combined_update",
                        "command": combined_cmd,
                        "error": (combined_result.stderr or combined_result.stdout).strip(),
                    }
                )
            else:
                info["processed"] += 1
                logger.log(f"Kernel export-all updated: {info['processed']}/{info['router_containers']} ({name})")
            summary.update_stage("start_kernel", **info)

            wait_for_system_idle(
                logger,
                "Inter-router kernel update wait",
                args.kernel_load_threshold,
                args.kernel_load_check_interval,
            )
            if args.kernel_per_router_delay > 0:
                time.sleep(args.kernel_per_router_delay)

    logger.log(
        "start-kernel finished: "
        f"processed={info['processed']} failed={info['failed']} "
        f"skipped_not_running={info['skipped_not_running']} "
        f"skipped_no_kernel_conf={info['skipped_no_kernel_conf']}"
    )

    if info["failed"] > 0:
        raise StageError(f"start-kernel failed for {info['failed']} containers", info)
    if info["processed"] == 0:
        raise StageError("start-kernel did not update any router containers", info)
    return info


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


def remote_command_args(args, remote_runner, remote_output, remote_summary, remote_pid_cache):
    cmd = [str(remote_runner), str(remote_output)]
    if args.project_name:
        cmd += ["--project-name", args.project_name]
    cmd += ["--build-mode", args.build_mode]
    cmd += ["--build-batch", str(args.build_batch)]
    cmd += ["--build-jobs", str(args.build_jobs)]
    cmd += ["--up-batch", str(args.up_batch)]
    cmd += ["--up-jobs", str(args.up_jobs)]
    cmd += ["--summary-path", str(remote_summary)]
    cmd += ["--pid-cache-file", str(remote_pid_cache)]
    cmd += ["--bird-start-command", args.bird_start_command]
    cmd += ["--start-bird-delay", str(args.start_bird_delay)]
    cmd += ["--verify-bird-delay", str(args.verify_bird_delay)]
    cmd += ["--verify-bird-jobs", str(args.verify_bird_jobs)]
    cmd += ["--kernel-load-threshold", str(args.kernel_load_threshold)]
    cmd += ["--kernel-load-check-interval", str(args.kernel_load_check_interval)]
    cmd += ["--kernel-per-router-delay", str(args.kernel_per_router_delay)]
    cmd += ["--kernel-config-mode", args.kernel_config_mode]
    cmd += ["--entry-method", args.entry_method]
    cmd += ["--start-stage-jobs", str(args.start_stage_jobs)]
    cmd += ["--nsenter-sudo", args.nsenter_sudo]
    cmd += ["--nsenter-timeout", str(args.nsenter_timeout)]
    if args.container_name_filter:
        cmd += ["--container-name-filter", args.container_name_filter]
    if args.down_first:
        cmd.append("--down-first")
    if args.no_build:
        cmd.append("--no-build")
    if args.no_up:
        cmd.append("--no-up")
    if args.no_start_bird:
        cmd.append("--no-start-bird")
    if args.verify_bird:
        cmd.append("--verify-bird")
    if args.no_start_kernel:
        cmd.append("--no-start-kernel")
    if args.pull:
        cmd.append("--pull")
    return cmd


def attach_vm_host_summary(local_summary_path, host_data):
    if local_summary_path.exists():
        with local_summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
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
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(local_summary_path)


def fetch_remote_summary(vm_user, vm_host, remote_summary, local_summary_path, attempts=3, delay=5):
    local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    test_cmd = ssh_base(vm_user, vm_host) + ["test", "-f", str(remote_summary)]
    for attempt in range(1, attempts + 1):
        if run_streamed(test_cmd, check=False) != 0:
            if attempt < attempts:
                time.sleep(delay)
                continue
            return False
        rc = run_streamed(
            rsync_base(vm_user, vm_host)
            + [f"{vm_user}@{vm_host}:{remote_summary}", str(local_summary_path)],
            check=False,
        )
        if rc == 0:
            return True
        if attempt < attempts:
            time.sleep(delay)
    return False


def run_on_vm(args, output_dir, requested_summary_path=None):
    vm_host = args.vm
    vm_user = args.vm_user
    if not args.project_name:
        args.project_name = default_project_name(output_dir)
    stable_runner = Path(args.vm_runner or f"/home/{vm_user}/bird/run_all.py")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_dir, remote_output, topology_size = remote_run_paths(
        output_dir,
        args.project_name,
        vm_user,
        args.vm_output_root,
        timestamp,
    )
    if requested_summary_path is None:
        local_summary_path = local_vm_summary_path(topology_size, timestamp)
    else:
        local_summary_path = requested_summary_path
    compatibility_summary_path = output_dir / "summary.json"

    remote_summary = remote_run_dir / "summary.json"
    local_remote_summary_path = local_summary_path.parent / "remote_summary.json"
    remote_pid_cache = remote_run_dir / DEFAULT_PID_CACHE_FILE
    remote_runner = remote_run_dir / stable_runner.name
    remote_manifest = remote_run_dir / "run_manifest.json"

    summary = SummaryRecorder(
        local_summary_path,
        {
            "mode": "vm-host",
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
            "remote_pid_cache_file": str(remote_pid_cache),
            "vm": {"host": vm_host, "user": vm_user, "runner": str(remote_runner), "stable_runner": str(stable_runner)},
            "project_name": args.project_name,
            "topology_services": topology_size,
            "build_mode": args.build_mode,
            "verify_bird": args.verify_bird,
            "kernel_config_mode": args.kernel_config_mode,
            "entry_method": args.entry_method,
            "start_stage_jobs": args.start_stage_jobs,
        },
    )

    print("SeedEMU run_all VM mode started.", flush=True)
    print(f"Local output: {output_dir}", flush=True)
    print(f"VM: {vm_user}@{vm_host}", flush=True)
    print(f"Remote run dir: {remote_run_dir}", flush=True)
    print(f"Remote output: {remote_output}", flush=True)
    print(f"Remote runner: {remote_runner}", flush=True)
    print(f"Compose project: {args.project_name}", flush=True)
    print(f"Topology services: {topology_size}", flush=True)
    print(f"Summary: {local_summary_path}", flush=True)
    if local_summary_path != compatibility_summary_path:
        print(f"Compatibility summary copy: {compatibility_summary_path}", flush=True)

    remote_error = None
    try:
        summary.start_stage(
            "sync",
            remote_run_dir=str(remote_run_dir),
            remote_output_dir=str(remote_output),
            remote_runner=str(remote_runner),
        )
        run_streamed(ssh_base(vm_user, vm_host) + ["mkdir", "-p", str(remote_run_dir), str(remote_output), str(stable_runner.parent)])
        run_streamed(
            rsync_base(vm_user, vm_host)
            + [str(Path(__file__).resolve()), f"{vm_user}@{vm_host}:{stable_runner}"]
        )
        run_streamed(
            rsync_base(vm_user, vm_host)
            + [str(Path(__file__).resolve()), f"{vm_user}@{vm_host}:{remote_runner}"]
        )
        chmod_targets = [str(stable_runner), str(remote_runner)]
        helper_path = None
        if args.build_mode == "fast" and not args.no_build:
            helper_path = resolve_fast_builder_script()
            stable_helper = stable_runner.parent / helper_path.name
            remote_helper = remote_run_dir / helper_path.name
            run_streamed(
                rsync_base(vm_user, vm_host)
                + [str(helper_path), f"{vm_user}@{vm_host}:{stable_helper}"]
            )
            run_streamed(
                rsync_base(vm_user, vm_host)
                + [str(helper_path), f"{vm_user}@{vm_host}:{remote_helper}"]
            )
            chmod_targets.extend([str(stable_helper), str(remote_helper)])
        run_streamed(ssh_base(vm_user, vm_host) + ["chmod", "+x"] + chmod_targets)
        manifest = {
            "schema_version": 1,
            "created_at": now_iso(),
            "mode": "run_all.py vm-host",
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
            "remote_pid_cache_file": str(remote_pid_cache),
            "remote_runner": str(remote_runner),
            "stable_runner": str(stable_runner),
            "fast_builder_helper": str(helper_path) if helper_path else None,
            "project_name": args.project_name,
            "topology_services": topology_size,
            "build_mode": args.build_mode,
            "verify_bird": args.verify_bird,
            "verify_bird_delay": args.verify_bird_delay,
            "verify_bird_jobs": args.verify_bird_jobs,
            "kernel_config_mode": args.kernel_config_mode,
            "entry_method": args.entry_method,
            "start_stage_jobs": args.start_stage_jobs,
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
        remote_cmd = " ".join(
            shlex.quote(part)
            for part in remote_command_args(args, remote_runner, remote_output, remote_summary, remote_pid_cache)
        )
        rc = run_streamed(ssh_base(vm_user, vm_host) + [remote_cmd], check=False)
        if rc != 0:
            remote_error = f"Remote run_all exited with code {rc}"
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
        if local_summary_path != compatibility_summary_path:
            compatibility_summary_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_summary_path, compatibility_summary_path)

    if remote_error:
        raise RuntimeError(remote_error)
    print("SeedEMU run_all VM mode completed successfully.", flush=True)


def run_stage(summary, name, func, *args, **kwargs):
    summary.start_stage(name)
    try:
        info = func(*args, **kwargs)
        summary.finish_stage(name, True, **(info or {}))
        return info
    except StageError as exc:
        summary.finish_stage(name, False, error=str(exc), **exc.details)
        raise
    except Exception as exc:
        summary.finish_stage(name, False, error=str(exc))
        raise


def run_local(args, output_dir, summary_path):
    project_name = args.project_name or default_project_name(output_dir)
    run_cwd = output_dir.parent
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / "run_logs" / f"seedemu_run_all_{timestamp}.log"
    logger = Logger(log_path)
    summary = SummaryRecorder(
        summary_path,
        {
            "mode": "local",
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
                "down_first": args.down_first,
                "no_build": args.no_build,
                "no_up": args.no_up,
                "no_start_bird": args.no_start_bird,
                "verify_bird": args.verify_bird,
                "verify_bird_delay": args.verify_bird_delay,
                "verify_bird_jobs": args.verify_bird_jobs,
                "no_start_kernel": args.no_start_kernel,
                "container_name_filter": args.container_name_filter,
                "start_bird_delay": args.start_bird_delay,
                "kernel_load_threshold": args.kernel_load_threshold,
                "kernel_load_check_interval": args.kernel_load_check_interval,
                "kernel_per_router_delay": args.kernel_per_router_delay,
                "kernel_config_mode": args.kernel_config_mode,
                "entry_method": args.entry_method,
                "start_stage_jobs": args.start_stage_jobs,
                "nsenter_sudo": args.nsenter_sudo,
                "nsenter_timeout": args.nsenter_timeout,
            },
        },
    )

    try:
        logger.log("SeedEMU run_all started.")
        logger.log(f"Output directory: {output_dir}")
        logger.log(f"Project name: {project_name}")
        logger.log(f"Run log: {log_path}")
        logger.log(f"Summary: {summary_path}")

        base_cmd = compose_cmd(output_dir, project_name)
        services = get_services(base_cmd, run_cwd, logger)
        summary.update(services={"count": len(services), "first": services[0] if services else None})
        logger.log(f"Compose services: {len(services)}")

        if args.down_first:
            run_stage(summary, "down", lambda: run_logged(base_cmd + ["down", "--remove-orphans"], run_cwd, logger) or {})

        if args.pull:
            logger.log("--pull is ignored by the local build flow.")

        if args.no_start_kernel:
            summary.skip_stage("prepare_kernel", "--no-start-kernel")
        elif args.kernel_config_mode == "build-context":
            run_stage(
                summary,
                "prepare_kernel",
                prepare_kernel_context_flow,
                output_dir,
                logger,
                args,
                summary,
            )
        else:
            summary.skip_stage("prepare_kernel", "--kernel-config-mode runtime")

        if args.no_build:
            summary.skip_stage("build", "--no-build", services_total=len(services))
        elif args.build_mode == "fast":
            run_stage(
                summary,
                "build",
                run_fastdocker_flow,
                base_cmd,
                output_dir,
                run_cwd,
                logger,
                project_name,
                summary,
            )
        else:
            run_stage(
                summary,
                "build",
                run_buildtest1_flow,
                base_cmd,
                output_dir,
                run_cwd,
                logger,
                args.build_batch,
                args.build_jobs,
                summary,
            )

        if args.no_up:
            summary.skip_stage("up", "--no-up", services_total=len(services))
        else:
            run_stage(
                summary,
                "up",
                run_dockerup_flow,
                base_cmd,
                run_cwd,
                logger,
                args.up_batch,
                args.up_jobs,
                summary,
            )

        if args.no_start_bird:
            summary.skip_stage("start_bird", "--no-start-bird")
        else:
            pid_cache_file = resolve_pid_cache_file(args.pid_cache_file, run_cwd)
            run_stage(
                summary,
                "start_bird",
                start_bird_flow,
                project_name,
                run_cwd,
                logger,
                args,
                pid_cache_file,
                summary,
            )

        if args.no_start_bird:
            summary.skip_stage("verify_bird", "--no-start-bird")
        elif args.verify_bird:
            run_stage(
                summary,
                "verify_bird",
                verify_bird_flow,
                project_name,
                run_cwd,
                logger,
                args,
                summary,
            )
        else:
            summary.skip_stage("verify_bird", "--verify-bird not set")

        if args.no_start_kernel:
            summary.skip_stage("start_kernel", "--no-start-kernel")
        elif args.kernel_config_mode == "build-context":
            summary.skip_stage(
                "start_kernel",
                "kernel.conf export-all was prepared in build contexts before build",
                mode="build-context",
            )
        else:
            run_stage(
                summary,
                "start_kernel",
                start_kernel_flow,
                project_name,
                run_cwd,
                logger,
                args,
                summary,
            )

        run_logged(base_cmd + ["ps"], run_cwd, logger)
        logger.log("SeedEMU run_all completed successfully.")
        summary.finalize(True)
        return 0
    except Exception as exc:
        logger.log(f"ERROR: {exc}")
        summary.finalize(False, error=str(exc))
        return 1
    finally:
        logger.close()


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    if args.vm:
        try:
            summary_path = resolve_summary_path(args.summary_path, output_dir) if args.summary_path else None
            run_on_vm(args, output_dir, summary_path)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    summary_path = resolve_summary_path(args.summary_path, output_dir)
    return run_local(args, output_dir, summary_path)


if __name__ == "__main__":
    sys.exit(main())
