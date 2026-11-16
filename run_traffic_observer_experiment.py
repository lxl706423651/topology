#!/usr/bin/env python3
"""Run a complete SeedEMU + BIRD + traffic-observer ICMP experiment.

The default invocation builds and starts one compiled SeedEMU output directory,
starts BIRD, switches the BIRD kernel protocol, verifies both control-plane and
FIB gates, starts the traffic observer, and records five isolated ping groups.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable


ROUTER_NAME_RE = re.compile(r"^as(\d+)brd-r(\d+)")
REQUIRED_PACKET_FIELDS = (
    "type",
    "timestamp",
    "timestampNs",
    "containerId",
    "nodeName",
    "nodeIp",
    "sourceIp",
    "destIp",
    "ipProtocol",
    "sourceContainerId",
    "sourceNodeName",
    "sourceNodeIp",
    "destContainerId",
    "destNodeName",
    "destNodeIp",
)
STAGE_ORDER = (
    "build",
    "up",
    "start-bird",
    "kernel",
    "observer",
    "capture",
)
VISUALIZATION_REPO_URL = "https://github.com/cy-gitlab/seed-visualization.git"
OBSERVER_ONLY_COMPOSE = """services:
  seedmu_traffic_observer_service:
    container_name: seedmu_traffic_observer_service
    init: true
    image: handsonsecurity/seedemu-traffic-observer-service:3.0
    build:
      context: .
      dockerfile: traffic-observer-service/Dockerfile
    privileged: true
    pid: host
    network_mode: host
    environment:
      TRAFFIC_INTERFACES: ${TRAFFIC_INTERFACES:-}
      TRAFFIC_FALLBACK_INTERFACES: ${TRAFFIC_FALLBACK_INTERFACES:-docker0}
      TRAFFIC_ONLY_SEED_CONTAINERS: ${TRAFFIC_ONLY_SEED_CONTAINERS:-true}
      TRAFFIC_DISCOVERY_CONCURRENCY: ${TRAFFIC_DISCOVERY_CONCURRENCY:-32}
      TRAFFIC_DOCKER_API_TIMEOUT_SECONDS: ${TRAFFIC_DOCKER_API_TIMEOUT_SECONDS:-7200}
      TRAFFIC_FILTER: ${TRAFFIC_FILTER:-}
      TRAFFIC_CONTROL_ADDR: ${TRAFFIC_CONTROL_ADDR:-:19092}
      EMULATOR_SERVICE_TRAFFIC_URL: ${EMULATOR_SERVICE_TRAFFIC_URL:-}
      TRAFFIC_PCAP_ENABLED: ${TRAFFIC_PCAP_ENABLED:-false}
      TRAFFIC_PCAP_OUTPUT_DIR: ${TRAFFIC_PCAP_OUTPUT_DIR:-/data/pcap}
      TRAFFIC_LOG_TIMEZONE: ${TRAFFIC_LOG_TIMEZONE:-Asia/Shanghai}
    volumes:
      - /sys/kernel/debug:/sys/kernel/debug
      - /sys/fs/bpf:/sys/fs/bpf
      - /lib/modules:/lib/modules:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traffic-observer-service/pcap:/data/pcap
    restart: unless-stopped
"""


class ExperimentError(RuntimeError):
    """Raised when a required experiment gate fails."""


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        with self.lock:
            print(line, flush=True)
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_capture(
    command: list[str],
    logger: Logger,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
    announce: bool = True,
) -> subprocess.CompletedProcess[str]:
    if announce:
        logger.log(f"COMMAND cwd={cwd or Path.cwd()}: {shell_join(command)}")
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise ExperimentError(
            f"command failed rc={proc.returncode}: {shell_join(command)}\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
        )
    return proc


def run_logged(
    command: list[str],
    logger: Logger,
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.log(f"COMMAND cwd={cwd or Path.cwd()}: {shell_join(command)} > {log_path} 2>&1")
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if check and proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise ExperimentError(
            f"command failed rc={proc.returncode}: {shell_join(command)}; "
            f"log={log_path}\n{tail}"
        )
    return proc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_observer_artifact(source: Path, destination: Path, logger: Logger) -> None:
    """Copy a recorder artifact, including root-owned 0640 bind-mount files."""
    try:
        shutil.copy2(source, destination)
    except PermissionError:
        run_capture(
            ["sudo", "-n", "install", "-m", "0644", str(source), str(destination)],
            logger,
        )


def natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def project_name(output_dir: Path) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", output_dir.name.lower()).strip("-_")
    if not value:
        raise ExperimentError(f"cannot derive Compose project name from {output_dir}")
    return value


def require_commands(names: Iterable[str]) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise ExperimentError(f"missing required commands: {', '.join(missing)}")


def prepare_visualization_repo(args: argparse.Namespace, logger: Logger) -> dict[str, Any]:
    """Clone when absent, otherwise preserve the user's visualization checkout."""
    require_commands(("git",))
    cloned = False
    if not args.visualization_dir.exists():
        args.visualization_dir.parent.mkdir(parents=True, exist_ok=True)
        run_capture(
            ["git", "clone", VISUALIZATION_REPO_URL, str(args.visualization_dir)],
            logger,
            cwd=args.visualization_dir.parent,
            timeout=args.stage_timeout,
        )
        cloned = True
    if not (args.visualization_dir / ".git").is_dir():
        raise ExperimentError(
            f"visualization directory is not a Git clone: {args.visualization_dir}"
        )

    compose_path = args.visualization_dir / "docker-compose.yml"
    if not compose_path.is_file():
        raise ExperimentError(f"missing visualization Compose file: {compose_path}")
    logger.log(f"preserving existing visualization checkout and Compose: {compose_path}")

    pcap_dir = args.visualization_dir / "traffic-observer-service" / "pcap"
    pcap_dir.mkdir(parents=True, exist_ok=True)
    return {
        "repository_url": VISUALIZATION_REPO_URL,
        "path": str(args.visualization_dir),
        "cloned": cloned,
        "compose_changed": False,
        "compose_preserved": True,
        "pcap_dir": str(pcap_dir),
    }


def check_preflight(args: argparse.Namespace, logger: Logger) -> dict[str, Any]:
    require_commands(("bash", "curl", "docker", "git", "nsenter", "python3", "sudo", "xargs"))
    compose_file = args.output_dir / "docker-compose.yml"
    observer_compose = args.visualization_dir / "docker-compose.yml"
    required_files = (
        compose_file,
        observer_compose,
        args.topology_dir / "1buildFastDockerImages.py",
        args.topology_dir / "dockerUp.sh",
        args.topology_dir / "start_bird0130.py",
        args.topology_dir / "start_bird_kernel.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ExperimentError(f"missing required files: {missing}")

    run_capture(["docker", "info"], logger, timeout=30)
    run_capture(["sudo", "-n", "true"], logger, timeout=10)
    observer_services = run_capture(
        ["docker", "compose", "-f", str(observer_compose), "config", "--services"],
        logger,
        cwd=args.visualization_dir,
    ).stdout.split()
    if "seedmu_traffic_observer_service" not in observer_services:
        raise ExperimentError(
            "visualization Compose does not define seedmu_traffic_observer_service; "
            f"found={observer_services}"
        )

    commit = run_capture(
        ["git", "-C", str(args.visualization_dir), "rev-parse", "HEAD"], logger
    ).stdout.strip()
    running = run_capture(
        ["docker", "ps", "--format", "{{.Names}}"], logger
    ).stdout.splitlines()
    if running and args.start_stage == "build":
        raise ExperimentError(
            "a fresh full run requires no pre-existing running containers; "
            f"found first containers={running[:10]}"
        )

    return {
        "compose_file": str(compose_file),
        "observer_compose": str(observer_compose),
        "observer_commit": commit,
        "observer_services": observer_services,
        "initial_running_containers": running,
    }


def stage_enabled(start_stage: str, stage: str) -> bool:
    return STAGE_ORDER.index(stage) >= STAGE_ORDER.index(start_stage)


def build_simulator(args: argparse.Namespace, logger: Logger, run_dir: Path) -> dict[str, Any]:
    compose_file = args.output_dir / "docker-compose.yml"
    started = time.monotonic()
    run_logged(
        [
            "python3",
            str(args.topology_dir / "1buildFastDockerImages.py"),
            str(compose_file),
            "--log-dir",
            str(run_dir / "build"),
        ],
        logger,
        run_dir / "build" / "fast_image_build.console.log",
        cwd=args.topology_dir,
        timeout=args.stage_timeout,
    )
    summary_path = args.output_dir / ".fastDockerBuild" / "summary.json"
    build_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "success": True,
        "duration_seconds": round(time.monotonic() - started, 3),
        "summary_path": str(summary_path),
        "image_count": build_summary.get("imageCount"),
        "base_count": build_summary.get("baseCount"),
    }


def compose_services(compose_file: Path, compose_project: str, logger: Logger) -> list[str]:
    proc = run_capture(
        [
            "docker",
            "compose",
            "-p",
            compose_project,
            "-f",
            str(compose_file),
            "config",
            "--services",
        ],
        logger,
    )
    services = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not services:
        raise ExperimentError(f"no services found in {compose_file}")
    return services


def run_compose_batch(
    index: int,
    services: list[str],
    compose_file: Path,
    compose_project: str,
    args: argparse.Namespace,
    logger: Logger,
    run_dir: Path,
) -> tuple[int, int]:
    command = [
        "docker",
        "compose",
        "-p",
        compose_project,
        "-f",
        str(compose_file),
        "up",
        "-d",
        "--no-build",
        *services,
    ]
    env = os.environ.copy()
    env["COMPOSE_PARALLEL_LIMIT"] = str(args.compose_parallel_limit)
    proc = run_logged(
        command,
        logger,
        run_dir / "compose_up" / f"batch_{index:03d}.log",
        cwd=args.output_dir,
        env=env,
        timeout=args.stage_timeout,
        check=False,
    )
    return index, proc.returncode


def project_container_states(compose_project: str, logger: Logger) -> list[dict[str, str]]:
    proc = run_capture(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
            "--format",
            '{{json .}}\t{{json (.Label "com.docker.compose.service")}}',
        ],
        logger,
        announce=False,
    )
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row_json, service_json = line.rsplit("\t", 1)
        row = json.loads(row_json)
        service = json.loads(service_json)
        if service:
            row["Service"] = service
        rows.append(row)
    return rows


def wait_for_project_running(
    compose_project: str,
    expected: int,
    logger: Logger,
    timeout: float,
) -> tuple[list[dict[str, str]], float]:
    started = time.monotonic()
    while True:
        rows = project_container_states(compose_project, logger)
        running = [row for row in rows if row.get("State") == "running"]
        graceful_exits = [
            row
            for row in rows
            if row.get("State") == "exited" and row.get("Status", "").startswith("Exited (0)")
        ]
        ready = running + graceful_exits
        elapsed = time.monotonic() - started
        logger.log(
            f"Compose readiness project={compose_project}: "
            f"created={len(rows)}/{expected}, running={len(running)}, "
            f"graceful_exits={len(graceful_exits)}, ready={len(ready)}/{expected}, "
            f"waited={elapsed:.1f}s"
        )
        if len(rows) == expected and len(ready) == expected:
            return rows, elapsed
        if elapsed >= timeout:
            bad = [row for row in rows if row.get("State") != "running"][:20]
            raise ExperimentError(
                f"Compose project did not reach {expected} running containers; "
                f"created={len(rows)} running={len(running)} "
                f"graceful_exits={len(graceful_exits)} bad={bad}"
            )
        time.sleep(10)


def start_simulator(args: argparse.Namespace, logger: Logger, run_dir: Path) -> dict[str, Any]:
    compose_file = args.output_dir / "docker-compose.yml"
    compose_project = project_name(args.output_dir)
    services = compose_services(compose_file, compose_project, logger)
    batches = [
        services[index : index + args.compose_batch_size]
        for index in range(0, len(services), args.compose_batch_size)
    ]
    started = time.monotonic()
    docker_up_env = os.environ.copy()
    docker_up_env.update(
        {
            "DOCKER_UP_PARALLEL_JOBS": str(args.compose_jobs),
            "DOCKER_UP_BATCH_SIZE": str(args.compose_batch_size),
            "COMPOSE_PARALLEL_LIMIT": str(args.compose_parallel_limit),
            "EXP_LOG_DIR": str(run_dir),
        }
    )
    docker_up_script = args.topology_dir / "dockerUp.sh"
    up_proc = run_logged(
        ["bash", str(docker_up_script), str(compose_file)],
        logger,
        run_dir / "compose_up" / "dockerUp.console.log",
        cwd=args.topology_dir,
        env=docker_up_env,
        timeout=args.stage_timeout,
        check=False,
    )
    if up_proc.returncode != 0:
        logger.log(f"dockerUp.sh failed rc={up_proc.returncode}; retrying once")
        up_proc = run_logged(
            ["bash", str(docker_up_script), str(compose_file)],
            logger,
            run_dir / "compose_up" / "dockerUp.retry.console.log",
            cwd=args.topology_dir,
            env=docker_up_env,
            timeout=args.stage_timeout,
            check=False,
        )
    if up_proc.returncode != 0:
        raise ExperimentError(
            f"dockerUp.sh failed after retry rc={up_proc.returncode}; "
            f"log={run_dir / 'compose_up' / 'dockerUp.retry.console.log'}"
        )

    rows, ready_wait = wait_for_project_running(
        compose_project, len(services), logger, args.stage_timeout
    )
    return {
        "success": True,
        "project": compose_project,
        "docker_up_script": str(docker_up_script),
        "service_count": len(services),
        "batch_count": len(batches),
        "running_count": sum(row.get("State") == "running" for row in rows),
        "graceful_exit_count": sum(
            row.get("State") == "exited" and row.get("Status", "").startswith("Exited (0)")
            for row in rows
        ),
        "ready_wait_seconds": round(ready_wait, 3),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def wait_for_load(
    threshold: float,
    interval: float,
    timeout: float,
    logger: Logger,
    gate_name: str,
) -> tuple[float, float]:
    started = time.monotonic()
    while True:
        load1 = os.getloadavg()[0]
        waited = time.monotonic() - started
        logger.log(
            f"load gate {gate_name}: load1={load1:.2f}, target=<{threshold}, "
            f"waited={waited:.1f}s"
        )
        if load1 < threshold:
            return load1, waited
        if waited >= timeout:
            raise ExperimentError(
                f"load gate {gate_name} timed out: load1={load1:.2f} >= {threshold}"
            )
        time.sleep(interval)


def router_representatives(compose_project: str, logger: Logger) -> dict[int, dict[str, Any]]:
    proc = run_capture(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
            "--format",
            "{{.Names}}",
        ],
        logger,
    )
    grouped: dict[int, list[tuple[int, str]]] = {}
    for name in proc.stdout.splitlines():
        match = ROUTER_NAME_RE.match(name)
        if not match:
            continue
        asn = int(match.group(1))
        router_number = int(match.group(2))
        grouped.setdefault(asn, []).append((router_number, name))
    if not grouped:
        raise ExperimentError(f"no brdnode containers found for project {compose_project}")
    return {
        asn: {
            "asn": asn,
            "router_number": sorted(values, key=lambda item: (item[0], item[1]))[0][0],
            "name": sorted(values, key=lambda item: (item[0], item[1]))[0][1],
        }
        for asn, values in grouped.items()
    }


def check_protocol_sample(row: dict[str, Any], logger: Logger) -> dict[str, Any]:
    name = row["name"]
    protocols = run_capture(
        ["docker", "exec", name, "birdc", "show", "protocols"],
        logger,
        timeout=60,
        check=False,
    )
    route_count = run_capture(
        ["docker", "exec", name, "birdc", "show", "route", "count"],
        logger,
        timeout=60,
        check=False,
    )
    failures: list[str] = []
    parsed = 0
    if protocols.returncode != 0:
        failures.append(f"birdc show protocols rc={protocols.returncode}")
    else:
        for line in protocols.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0] in {"BIRD", "Name"}:
                continue
            parsed += 1
            protocol_type = parts[1].lower()
            state = parts[3].lower()
            if state != "up":
                failures.append(line)
            elif protocol_type == "bgp" and "Established" not in line:
                failures.append(line)
    if parsed == 0:
        failures.append("no protocol rows parsed")
    if route_count.returncode != 0:
        failures.append(f"birdc show route count rc={route_count.returncode}")
    return {
        "asn": row["asn"],
        "container": name,
        "success": not failures,
        "failed_protocol_lines": failures[:100],
        "route_count_output": route_count.stdout.strip(),
        "raw_protocol_output": protocols.stdout if failures else "",
        "stderr": (protocols.stderr + route_count.stderr)[-2000:] if failures else "",
    }


def verify_after_start_bird(
    compose_project: str,
    args: argparse.Namespace,
    logger: Logger,
) -> dict[str, Any]:
    reps = router_representatives(compose_project, logger)
    selected = [reps[asn] for asn in sorted(reps)[:10]]
    if len(selected) < 10:
        raise ExperimentError(f"need 10 AS samples, found {len(selected)}")
    started = time.monotonic()
    attempts = 0
    last_checks: list[dict[str, Any]] = []
    total_load_wait = 0.0
    last_load1 = 0.0
    while True:
        last_load1, waited = wait_for_load(
            args.load_threshold,
            args.load_check_interval,
            args.stage_timeout,
            logger,
            "verify_after_start_bird",
        )
        total_load_wait += waited
        attempts += 1
        last_checks = [check_protocol_sample(row, logger) for row in selected]
        failures = [row for row in last_checks if not row["success"]]
        logger.log(
            f"verify_after_start_bird attempt={attempts}: "
            f"pass={len(last_checks) - len(failures)}/{len(last_checks)}"
        )
        if not failures:
            return {
                "success": True,
                "load1": last_load1,
                "load_wait_seconds": round(total_load_wait, 3),
                "sample_count": len(selected),
                "selected_containers": [row["name"] for row in selected],
                "failed_samples": [],
                "attempts": attempts,
                "checks": last_checks,
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        if time.monotonic() - started >= args.stage_timeout:
            raise ExperimentError(
                "verify_after_start_bird failed: "
                + json.dumps(failures, ensure_ascii=False)[:12000]
            )
        time.sleep(args.protocol_retry_interval)


def start_bird_stage(
    args: argparse.Namespace,
    logger: Logger,
    run_dir: Path,
    compose_project: str,
) -> dict[str, Any]:
    started = time.monotonic()
    command = [
        "sudo",
        "-n",
        "env",
        f"EXP_LOG_DIR={run_dir}",
        "python3",
        str(args.topology_dir / "start_bird0130.py"),
    ]
    run_logged(
        command,
        logger,
        run_dir / "bird" / "start_bird0130.console.log",
        cwd=run_dir,
        timeout=args.stage_timeout,
    )
    verification = verify_after_start_bird(compose_project, args, logger)
    return {
        "success": True,
        "command": shell_join(command),
        "verify_after_start_bird": verification,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def parse_bird_network_count(output: str) -> tuple[int, str]:
    match = re.search(r"for\s+(\d+)\s+networks", output)
    if match:
        return int(match.group(1)), "networks"
    match = re.search(r"^\s*(\d+)\s+routes", output)
    if match:
        return int(match.group(1)), "routes_fallback"
    raise ExperimentError(f"cannot parse BIRD route count output: {output!r}")


def check_fib_sample(row: dict[str, Any], logger: Logger) -> dict[str, Any]:
    name = row["name"]
    bird = run_capture(
        ["docker", "exec", name, "birdc", "show", "route", "count"],
        logger,
        timeout=120,
        check=False,
    )
    fib = run_capture(
        ["docker", "exec", name, "sh", "-lc", "ip route | wc -l"],
        logger,
        timeout=120,
        check=False,
    )
    error = ""
    bird_count = -1
    parse_mode = ""
    fib_count = -1
    try:
        if bird.returncode != 0:
            raise ExperimentError(f"birdc rc={bird.returncode}: {bird.stderr[-500:]}")
        bird_count, parse_mode = parse_bird_network_count(bird.stdout)
        if fib.returncode != 0 or not fib.stdout.strip().isdigit():
            raise ExperimentError(f"invalid ip route count: {fib.stdout!r} {fib.stderr[-500:]}")
        fib_count = int(fib.stdout.strip())
    except Exception as exc:
        error = str(exc)
    success = not error and bird_count < fib_count
    if not success and not error:
        error = f"expected bird_network_count < ip_route_count, got {bird_count} >= {fib_count}"
    return {
        "asn": row["asn"],
        "container": name,
        "success": success,
        "bird_network_count": bird_count,
        "bird_parse_mode": parse_mode,
        "ip_route_count": fib_count,
        "error": error,
        "bird_raw": bird.stdout.strip() if not success else "",
        "fib_raw": fib.stdout.strip() if not success else "",
    }


def verify_after_fib_write(
    compose_project: str,
    args: argparse.Namespace,
    logger: Logger,
) -> dict[str, Any]:
    reps = router_representatives(compose_project, logger)
    selected = [reps[asn] for asn in sorted(reps)]
    started = time.monotonic()
    attempts = 0
    total_load_wait = 0.0
    last_load1 = 0.0
    while True:
        last_load1, waited = wait_for_load(
            args.load_threshold,
            args.load_check_interval,
            args.stage_timeout,
            logger,
            "verify_after_fib_write",
        )
        total_load_wait += waited
        attempts += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.verify_jobs) as executor:
            checks = list(executor.map(lambda row: check_fib_sample(row, logger), selected))
        failures = [row for row in checks if not row["success"]]
        logger.log(
            f"verify_after_fib_write attempt={attempts}: "
            f"pass={len(checks) - len(failures)}/{len(checks)}"
        )
        if not failures:
            return {
                "success": True,
                "load1": last_load1,
                "load_wait_seconds": round(total_load_wait, 3),
                "as_count": len(checks),
                "checked_containers": [row["name"] for row in selected],
                "failed_samples": [],
                "attempts": attempts,
                "checks": checks,
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        if time.monotonic() - started >= args.stage_timeout:
            raise ExperimentError(
                "verify_after_fib_write failed: "
                + json.dumps(failures[:50], ensure_ascii=False)[:12000]
            )
        logger.log(
            "FIB gate not ready; failed samples="
            + json.dumps(failures[:10], ensure_ascii=False)
        )
        time.sleep(args.fib_retry_interval)


def kernel_stage(
    args: argparse.Namespace,
    logger: Logger,
    run_dir: Path,
    compose_project: str,
) -> dict[str, Any]:
    started = time.monotonic()
    command = [
        "sudo",
        "-n",
        "env",
        f"EXP_LOG_DIR={run_dir}",
        "python3",
        str(args.topology_dir / "start_bird_kernel.py"),
    ]
    run_logged(
        command,
        logger,
        run_dir / "kernel" / "start_bird_kernel.console.log",
        cwd=run_dir,
        timeout=args.stage_timeout,
    )
    verification = verify_after_fib_write(compose_project, args, logger)
    return {
        "success": True,
        "command": shell_join(command),
        "verify_after_fib_write": verification,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def api_json(
    method: str,
    path: str,
    logger: Logger,
    *,
    body: dict[str, Any] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    command = ["curl", "--fail", "--silent", "--show-error", "-X", method]
    if body is not None:
        command.extend(
            [
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(body, separators=(",", ":")),
            ]
        )
    command.append(f"http://127.0.0.1:19092{path}")
    proc = run_capture(command, logger, timeout=30, check=check)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON from {path}: {proc.stdout!r}") from exc


def observer_stage(args: argparse.Namespace, logger: Logger, run_dir: Path) -> dict[str, Any]:
    compose = args.visualization_dir / "docker-compose.yml"
    started = time.monotonic()
    build_command = [
        "docker",
        "compose",
        "-f",
        str(compose),
        "build",
        "seedmu_traffic_observer_service",
    ]
    run_logged(
        build_command,
        logger,
        run_dir / "observer" / "build.log",
        cwd=args.visualization_dir,
        timeout=args.stage_timeout,
    )
    up_command = [
        "docker",
        "compose",
        "-f",
        str(compose),
        "up",
        "-d",
        "--no-deps",
        "seedmu_traffic_observer_service",
    ]
    run_logged(
        up_command,
        logger,
        run_dir / "observer" / "up.log",
        cwd=args.visualization_dir,
        timeout=600,
    )

    ready_started = time.monotonic()
    last_error = ""
    while time.monotonic() - ready_started < args.observer_ready_timeout:
        proc = run_capture(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "http://127.0.0.1:19092/filter",
            ],
            logger,
            timeout=30,
            check=False,
            announce=False,
        )
        if proc.returncode == 0:
            try:
                filter_status = json.loads(proc.stdout)
                break
            except json.JSONDecodeError:
                last_error = proc.stdout
        else:
            last_error = proc.stderr
        state = run_capture(
            ["docker", "inspect", "--format", "{{.State.Status}}", "seedmu_traffic_observer_service"],
            logger,
            timeout=30,
            check=False,
            announce=False,
        )
        logger.log(
            f"observer readiness: container_state={state.stdout.strip() or 'missing'}, "
            f"waited={time.monotonic() - ready_started:.1f}s"
        )
        time.sleep(10)
    else:
        final_probe = run_capture(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "http://127.0.0.1:19092/filter",
            ],
            logger,
            timeout=30,
            check=False,
        )
        if final_probe.returncode == 0:
            filter_status = json.loads(final_probe.stdout)
        else:
            logs = run_capture(
                ["docker", "logs", "--tail", "200", "seedmu_traffic_observer_service"],
                logger,
                check=False,
            )
            raise ExperimentError(
                f"observer API did not become ready: {last_error}; logs={logs.stdout[-8000:]}"
                f"{logs.stderr[-8000:]}"
            )

    api_json("PUT", "/pcap", logger, body={"enabled": False})
    filter_status = api_json("PUT", "/filter", logger, body={"filter": "icmp"})
    return {
        "success": True,
        "build_command": shell_join(build_command),
        "up_command": shell_join(up_command),
        "api_url": "http://127.0.0.1:19092",
        "filter": filter_status,
        "ready_wait_seconds": round(time.monotonic() - ready_started, 3),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def running_router_details(compose_project: str, logger: Logger) -> list[dict[str, Any]]:
    proc = run_capture(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
            "--format",
            "{{.Names}}",
        ],
        logger,
    )
    names = sorted(
        [name for name in proc.stdout.splitlines() if ROUTER_NAME_RE.match(name)],
        key=natural_key,
    )
    if not names:
        raise ExperimentError("no running router containers available for ping selection")
    inspect = run_capture(["docker", "inspect", *names], logger, timeout=300)
    rows = []
    for item in json.loads(inspect.stdout):
        name = item["Name"].lstrip("/")
        match = ROUTER_NAME_RE.match(name)
        if not match:
            continue
        networks = []
        for network_name, network in item["NetworkSettings"]["Networks"].items():
            address = network.get("IPAddress", "")
            if address:
                networks.append(
                    {
                        "network": network_name,
                        "ip": address,
                        "mac": network.get("MacAddress", ""),
                    }
                )
        rows.append(
            {
                "id": item["Id"],
                "name": name,
                "asn": int(match.group(1)),
                "router_number": int(match.group(2)),
                "networks": networks,
            }
        )
    return rows


def select_ping_pairs(
    compose_project: str,
    count: int,
    logger: Logger,
    selection_path: Path,
    min_hops: int,
    max_hops: int,
    probe_timeout: int,
    probe_limit: int,
    requested_pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    routers = running_router_details(compose_project, logger)
    router_by_name = {router["name"]: router for router in routers}
    by_network: dict[str, list[str]] = {}
    for router in routers:
        for network in router["networks"]:
            by_network.setdefault(network["network"], []).append(router["name"])

    adjacency = {router["name"]: set() for router in routers}
    for members in by_network.values():
        for name in members:
            adjacency[name].update(member for member in members if member != name)

    def routed_endpoint(router: dict[str, Any]) -> dict[str, Any]:
        routed_networks = sorted(
            (
                network
                for network in router["networks"]
                if "_net_ix_" not in network["network"]
            ),
            key=lambda network: natural_key(network["network"]),
        )
        if not routed_networks:
            raise ExperimentError(
                f"router has no non-IX routed endpoint: {router['name']}"
            )
        network = routed_networks[0]
        return {
            "id": router["id"],
            "name": router["name"],
            "asn": router["asn"],
            "router_number": router["router_number"],
            "network": network["network"],
            "ip": network["ip"],
            "mac": network["mac"],
        }

    endpoints = {
        name: routed_endpoint(router) for name, router in router_by_name.items()
    }

    def distances_from(source_name: str) -> dict[str, int]:
        distances = {source_name: 0}
        queue = [source_name]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def candidate(
        source_name: str, destination_name: str, graph_hops: int
    ) -> dict[str, Any]:
        source_networks = {
            network["network"] for network in router_by_name[source_name]["networks"]
        }
        destination_networks = {
            network["network"]
            for network in router_by_name[destination_name]["networks"]
        }
        return {
            "graph_hops": graph_hops,
            "shared_networks": sorted(
                source_networks & destination_networks, key=natural_key
            ),
            "source": endpoints[source_name],
            "destination": endpoints[destination_name],
        }

    def probe_candidate(pair: dict[str, Any]) -> dict[str, Any]:
        source = pair["source"]
        destination = pair["destination"]
        base_command = [
            "docker",
            "exec",
            source["name"],
            "ping",
            "-n",
            "-I",
            source["ip"],
            "-c",
            "1",
            "-W",
            str(probe_timeout),
        ]
        result: dict[str, Any] = {
            "success": False,
            "source": source["name"],
            "sourceIp": source["ip"],
            "destination": destination["name"],
            "destinationIp": destination["ip"],
            "graphHops": pair["graph_hops"],
            "ttlProbes": [],
        }

        source_route = run_capture(
            [
                "docker",
                "exec",
                source["name"],
                "ip",
                "route",
                "get",
                destination["ip"],
                "from",
                source["ip"],
            ],
            logger,
            check=False,
        )
        source_route_output = source_route.stdout + source_route.stderr
        result["sourceRouteReturncode"] = source_route.returncode
        result["sourceRouteOutput"] = source_route_output[-2000:]
        if source_route.returncode != 0:
            result["reason"] = "source FIB has no route to destination"
            return result

        reverse_route = run_capture(
            [
                "docker",
                "exec",
                destination["name"],
                "ip",
                "route",
                "get",
                source["ip"],
                "from",
                destination["ip"],
            ],
            logger,
            check=False,
        )
        reverse_route_output = reverse_route.stdout + reverse_route.stderr
        result["reverseRouteReturncode"] = reverse_route.returncode
        result["reverseRouteOutput"] = reverse_route_output[-2000:]
        if reverse_route.returncode != 0:
            result["reason"] = "destination FIB has no return route to source"
            return result

        reachability = run_capture(
            [*base_command, destination["ip"]], logger, check=False
        )
        reachability_output = reachability.stdout + reachability.stderr
        result["reachabilityReturncode"] = reachability.returncode
        result["reachabilityOutput"] = reachability_output[-2000:]
        if reachability.returncode != 0:
            result["reason"] = "ordinary ping did not reach destination"
            return result

        measured_hops = None
        for ttl in range(1, max_hops + 1):
            proc = run_capture(
                [*base_command, "-t", str(ttl), destination["ip"]],
                logger,
                check=False,
            )
            output = proc.stdout + proc.stderr
            ttl_exceeded = re.search(r"time to live exceeded", output, re.IGNORECASE)
            hop_match = re.search(
                r"From\s+([0-9]+(?:\.[0-9]+){3}).*time to live exceeded",
                output,
                re.IGNORECASE,
            )
            probe = {
                "ttl": ttl,
                "returncode": proc.returncode,
                "response": (
                    "destination"
                    if proc.returncode == 0
                    else "ttl-exceeded"
                    if ttl_exceeded
                    else "no-response"
                ),
                "hopIp": hop_match.group(1) if hop_match else None,
                "output": output[-1000:],
            }
            result["ttlProbes"].append(probe)
            if proc.returncode == 0:
                measured_hops = ttl
                break

        result["measuredHops"] = measured_hops
        if measured_hops is None:
            result["reason"] = f"destination was not reached within {max_hops} TTL hops"
            return result
        if measured_hops < min_hops:
            result["reason"] = (
                f"measured path is only {measured_hops} hops; minimum is {min_hops}"
            )
            return result
        result["success"] = True
        return result

    selection = {
        "success": False,
        "minimum_hops": min_hops,
        "maximum_hops": max_hops,
        "probe_timeout_seconds": probe_timeout,
        "probe_limit": probe_limit,
        "router_count": len(routers),
        "candidate_count": 0,
        "attempts": [],
        "selected": [],
    }

    def persist_selection() -> None:
        write_json(selection_path, selection)

    if requested_pairs:
        used_containers: set[str] = set()
        used_all_asns: set[int] = set()
        selected: list[dict[str, Any]] = []
        for source_name, destination_name in requested_pairs:
            if source_name not in router_by_name or destination_name not in router_by_name:
                raise ExperimentError(
                    "requested ping container is not running: "
                    f"source={source_name!r} destination={destination_name!r}"
                )
            source_router = router_by_name[source_name]
            destination_router = router_by_name[destination_name]
            if source_name in used_containers or destination_name in used_containers:
                raise ExperimentError("requested ping pairs reuse a container")
            source_asn = source_router["asn"]
            destination_asn = destination_router["asn"]
            if source_asn in used_all_asns or destination_asn in used_all_asns:
                raise ExperimentError("requested ping pairs must use ten globally distinct ASNs")
            graph_hops = distances_from(source_name).get(destination_name)
            if graph_hops is None or graph_hops < min_hops:
                raise ExperimentError(
                    "requested ping endpoints do not have the required physical path: "
                    f"source={source_name} destination={destination_name} "
                    f"graph_hops={graph_hops} minimum={min_hops}"
                )
            pair = candidate(source_name, destination_name, graph_hops)
            probe = probe_candidate(pair)
            selection["attempts"].append(probe)
            persist_selection()
            if not probe["success"]:
                raise ExperimentError(
                    "requested ping endpoints failed hop verification: "
                    f"source={source_name} destination={destination_name} "
                    f"reason={probe.get('reason')}"
                )
            pair["group"] = len(selected) + 1
            pair["hop_probe"] = probe
            selected.append(pair)
            used_containers.update((source_name, destination_name))
            used_all_asns.update((source_asn, destination_asn))
        if len(selected) != count:
            raise ExperimentError(
                f"expected {count} requested ping pairs, selected={len(selected)}"
            )
        selection["success"] = True
        selection["selected"] = selected
        persist_selection()
        logger.log(
            f"selected requested multihop ping groups: "
            f"{json.dumps(selected, ensure_ascii=False)}"
        )
        return selected

    best_by_as_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    names = sorted(router_by_name, key=natural_key)
    name_positions = {name: index for index, name in enumerate(names)}
    for source_name in names:
        source_asn = router_by_name[source_name]["asn"]
        for destination_name, graph_hops in distances_from(source_name).items():
            if name_positions[destination_name] <= name_positions[source_name]:
                continue
            destination_asn = router_by_name[destination_name]["asn"]
            if source_asn == destination_asn or graph_hops < min_hops:
                continue
            pair = candidate(source_name, destination_name, graph_hops)
            as_pair = tuple(sorted((source_asn, destination_asn)))
            values = best_by_as_pair.setdefault(as_pair, [])
            values.append(pair)
            values.sort(
                key=lambda value: (
                    -value["graph_hops"],
                    natural_key(value["source"]["name"]),
                    natural_key(value["destination"]["name"]),
                )
            )
            del values[3:]

    candidates = [pair for values in best_by_as_pair.values() for pair in values]
    candidates.sort(
        key=lambda value: (
            -value["graph_hops"],
            value["source"]["asn"],
            value["destination"]["asn"],
            natural_key(value["source"]["name"]),
            natural_key(value["destination"]["name"]),
        )
    )
    selection["candidate_count"] = len(candidates)
    persist_selection()

    used_all_asns: set[int] = set()
    selected: list[dict[str, Any]] = []
    for pair in candidates:
        if len(selected) >= count or len(selection["attempts"]) >= probe_limit:
            break
        source_asn = pair["source"]["asn"]
        destination_asn = pair["destination"]["asn"]
        if source_asn in used_all_asns or destination_asn in used_all_asns:
            continue
        probe = probe_candidate(pair)
        selection["attempts"].append(probe)
        persist_selection()
        if not probe["success"]:
            continue
        pair["group"] = len(selected) + 1
        pair["hop_probe"] = probe
        selected.append(pair)
        used_all_asns.update((source_asn, destination_asn))
        selection["selected"] = selected
        persist_selection()

    if len(selected) != count:
        selection["selected"] = selected
        persist_selection()
        raise ExperimentError(
            f"could not select {count} reachable ping groups with ten globally "
            f"distinct ASNs and at least {min_hops} hops after "
            f"{len(selection['attempts'])} probes; selected={selected}"
        )
    selection["success"] = True
    selection["selected"] = selected
    persist_selection()
    logger.log(f"selected multihop ping groups: {json.dumps(selected, ensure_ascii=False)}")
    return selected


def packet_has_required_fields(packet: dict[str, Any]) -> bool:
    return all(field in packet and packet[field] not in (None, "") for field in REQUIRED_PACKET_FIELDS)


def observed_path(
    messages: list[dict[str, Any]], logger: Logger
) -> list[dict[str, Any]]:
    path = []
    seen: set[str] = set()
    for message in sorted(messages, key=lambda value: int(value.get("timestampNs", 0))):
        container_id = message.get("containerId", "")
        if not container_id or container_id in seen:
            continue
        seen.add(container_id)
        path.append(
            {
                "containerId": container_id,
                "nodeName": message.get("nodeName"),
                "nodeIp": message.get("nodeIp"),
            }
        )
    if not path:
        return path

    inspect = run_capture(
        ["docker", "inspect", *[hop["containerId"] for hop in path]], logger
    )
    metadata = {}
    for item in json.loads(inspect.stdout):
        labels = item.get("Config", {}).get("Labels") or {}
        metadata[item["Id"]] = {
            "containerName": item.get("Name", "").lstrip("/"),
            "asn": labels.get("org.seedsecuritylabs.seedemu.meta.asn"),
            "role": labels.get("org.seedsecuritylabs.seedemu.meta.role"),
        }
    for hop in path:
        hop.update(metadata.get(hop["containerId"], {}))
    return path


def capture_ping_group(
    pair: dict[str, Any],
    args: argparse.Namespace,
    logger: Logger,
    run_dir: Path,
) -> dict[str, Any]:
    group_number = pair["group"]
    group_dir = run_dir / "captures" / f"group_{group_number:02d}"
    group_dir.mkdir(parents=True, exist_ok=True)
    source = pair["source"]
    destination = pair["destination"]

    api_json("PUT", "/pcap", logger, body={"enabled": False})
    # The observer opens its output files lazily when the first matching packet
    # arrives.  Do not require file paths from the enable response itself.
    api_json("PUT", "/pcap", logger, body={"enabled": True})

    ping_command = [
        "docker",
        "exec",
        source["name"],
        "ping",
        "-n",
        "-I",
        source["ip"],
        "-c",
        str(args.ping_count),
        "-W",
        str(args.ping_timeout),
        destination["ip"],
    ]
    ping_log = group_dir / "ping.log"
    ping_log.write_text("", encoding="utf-8")
    ping_proc = run_logged(
        ping_command,
        logger,
        ping_log,
        timeout=args.ping_count * args.ping_timeout + 60,
        check=False,
    )
    time.sleep(args.capture_settle_seconds)
    active_status = api_json("GET", "/pcap", logger)
    api_json("PUT", "/pcap", logger, body={"enabled": False})
    if ping_proc.returncode != 0:
        raise ExperimentError(
            f"group {group_number}: ping failed rc={ping_proc.returncode}; "
            f"log={ping_log}"
        )

    pcap_container_path = active_status.get("currentPcapFile", "")
    json_container_path = active_status.get("currentJsonFile", "")
    if not pcap_container_path or not json_container_path:
        raise ExperimentError(
            f"group {group_number}: observer produced no output paths; "
            f"status={active_status}"
        )
    if int(active_status.get("packetCount", 0)) <= 0 or int(active_status.get("messageCount", 0)) <= 0:
        raise ExperimentError(
            f"group {group_number}: empty observer session status={active_status}"
        )

    observer_host_dir = args.visualization_dir / "traffic-observer-service" / "pcap"
    source_pcap = observer_host_dir / Path(pcap_container_path).name
    source_json = observer_host_dir / Path(json_container_path).name
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (not source_pcap.is_file() or not source_json.is_file()):
        time.sleep(1)
    if not source_pcap.is_file() or not source_json.is_file():
        raise ExperimentError(
            f"group {group_number}: recorded files missing: pcap={source_pcap}, json={source_json}"
        )

    copied_pcap = group_dir / "packets.pcap"
    copied_json = group_dir / "packets.raw.json"
    copy_observer_artifact(source_pcap, copied_pcap, logger)
    copy_observer_artifact(source_json, copied_json, logger)
    messages = json.loads(copied_json.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise ExperimentError(f"group {group_number}: recorded JSON is not an array")
    relevant = [
        message
        for message in messages
        if message.get("type") == "packet"
        and message.get("ipProtocol") == "icmp"
        and (
            (
                message.get("sourceIp") == source["ip"]
                and message.get("destIp") == destination["ip"]
            )
            or (
                message.get("sourceIp") == destination["ip"]
                and message.get("destIp") == source["ip"]
            )
        )
    ]
    forward_samples = [
        message
        for message in relevant
        if message.get("sourceIp") == source["ip"]
        and message.get("destIp") == destination["ip"]
        and message.get("sourceContainerId") == source["id"]
        and message.get("destContainerId") == destination["id"]
        and packet_has_required_fields(message)
    ]
    reverse_samples = [
        message
        for message in relevant
        if message.get("sourceIp") == destination["ip"]
        and message.get("destIp") == source["ip"]
        and message.get("sourceContainerId") == destination["id"]
        and message.get("destContainerId") == source["id"]
        and packet_has_required_fields(message)
    ]
    if not forward_samples:
        raise ExperimentError(
            f"group {group_number}: no complete forward packet metadata matched selected endpoints; "
            f"relevant_count={len(relevant)}"
        )
    if not reverse_samples:
        raise ExperimentError(
            f"group {group_number}: no complete reverse packet metadata matched selected endpoints; "
            f"relevant_count={len(relevant)}"
        )

    forward_path = observed_path(forward_samples, logger)
    reverse_path = observed_path(reverse_samples, logger)
    minimum_direction_records = args.ping_count * args.min_ping_hops
    if len(forward_samples) < minimum_direction_records or len(forward_path) < args.min_ping_hops:
        raise ExperimentError(
            f"group {group_number}: observer did not capture at least "
            f"{args.min_ping_hops} forward hops per flow; "
            f"forward_records={len(forward_samples)} unique_nodes={len(forward_path)} "
            f"required_records={minimum_direction_records}"
        )
    if len(reverse_samples) < minimum_direction_records or len(reverse_path) < args.min_ping_hops:
        raise ExperimentError(
            f"group {group_number}: observer did not capture at least "
            f"{args.min_ping_hops} reverse hops per flow; "
            f"reverse_records={len(reverse_samples)} unique_nodes={len(reverse_path)} "
            f"required_records={minimum_direction_records}"
        )
    if copied_pcap.stat().st_size <= 24:
        raise ExperimentError(f"group {group_number}: pcap has no packet records")

    filtered_path = group_dir / "packets.filtered.json"
    sample_path = group_dir / "packet.sample.json"
    metadata_path = group_dir / "group.json"
    write_json(filtered_path, relevant)
    write_json(sample_path, forward_samples[0])
    result = {
        **pair,
        "success": True,
        "ping_command": shell_join(ping_command),
        "observer_status": active_status,
        "raw_message_count": len(messages),
        "relevant_message_count": len(relevant),
        "forward_message_count": len(forward_samples),
        "reverse_message_count": len(reverse_samples),
        "observed_forward_hops": len(forward_path),
        "observed_reverse_hops": len(reverse_path),
        "observed_forward_path": forward_path,
        "observed_reverse_path": reverse_path,
        "pcap_bytes": copied_pcap.stat().st_size,
        "artifacts": {
            "pcap": str(copied_pcap),
            "raw_json": str(copied_json),
            "filtered_json": str(filtered_path),
            "sample_json": str(sample_path),
            "ping_log": str(ping_log),
        },
        "sample": forward_samples[0],
    }
    write_json(metadata_path, result)
    return result


def capture_stage(
    args: argparse.Namespace,
    logger: Logger,
    run_dir: Path,
    compose_project: str,
) -> dict[str, Any]:
    started = time.monotonic()
    pairs = select_ping_pairs(
        compose_project,
        args.ping_groups,
        logger,
        run_dir / "captures" / "pair_selection.json",
        args.min_ping_hops,
        args.max_ping_hops,
        args.pair_probe_timeout,
        args.pair_probe_limit,
        requested_pairs=args.ping_pair,
    )
    filter_status = api_json("PUT", "/filter", logger, body={"filter": "icmp"})
    results = []
    try:
        for pair in pairs:
            result = capture_ping_group(pair, args, logger, run_dir)
            results.append(result)
            logger.log(
                f"capture group {pair['group']} passed: "
                f"{pair['source']['name']} ({pair['source']['ip']}) -> "
                f"{pair['destination']['name']} ({pair['destination']['ip']}), "
                f"packets={result['relevant_message_count']}"
            )
    finally:
        api_json("PUT", "/pcap", logger, body={"enabled": False}, check=False)
        api_json("PUT", "/filter", logger, body={"filter": ""}, check=False)

    all_packets: list[dict[str, Any]] = []
    samples = []
    for result in results:
        filtered = json.loads(Path(result["artifacts"]["filtered_json"]).read_text(encoding="utf-8"))
        all_packets.extend(filtered)
        samples.append(result["sample"])
    jsonl_path = run_dir / "captures" / "packets_all.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for packet in all_packets:
            handle.write(json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n")
    samples_path = run_dir / "captures" / "packet_samples.json"
    manifest_path = run_dir / "captures" / "capture_manifest.json"
    write_json(samples_path, samples)
    write_json(manifest_path, results)
    return {
        "success": len(results) == args.ping_groups,
        "filter_status": filter_status,
        "group_count": len(results),
        "groups": results,
        "combined_packet_count": len(all_packets),
        "artifacts": {
            "manifest": str(manifest_path),
            "samples": str(samples_path),
            "jsonl": str(jsonl_path),
        },
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    default_topology = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_topology / "output_1078",
        help="Compiled SeedEMU output directory containing docker-compose.yml.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("/home/lxl/seed-visualization"),
        help="Clone target; cloned automatically and reduced to observer-only Compose.",
    )
    parser.add_argument("--topology-dir", type=Path, default=default_topology)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--start-stage", choices=STAGE_ORDER, default="build")
    parser.add_argument("--compose-jobs", type=int, default=8)
    parser.add_argument("--compose-batch-size", type=int, default=50)
    parser.add_argument("--compose-parallel-limit", type=int, default=50)
    parser.add_argument("--verify-jobs", type=int, default=32)
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-check-interval", type=float, default=30.0)
    parser.add_argument("--protocol-retry-interval", type=float, default=30.0)
    parser.add_argument("--fib-retry-interval", type=float, default=30.0)
    parser.add_argument("--stage-timeout", type=float, default=14400.0)
    parser.add_argument(
        "--observer-ready-timeout",
        type=float,
        default=7200.0,
        help="Maximum seconds to wait for observer interface discovery and API readiness.",
    )
    parser.add_argument("--ping-groups", type=int, default=5)
    parser.add_argument(
        "--ping-pair",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="Explicit container pair; repeat exactly five times. Defaults to automatic distinct-AS selection.",
    )
    parser.add_argument("--ping-count", type=int, default=5)
    parser.add_argument("--ping-timeout", type=int, default=2)
    parser.add_argument(
        "--min-ping-hops",
        type=int,
        default=5,
        help="Require every selected forward and reverse path to contain at least this many hops.",
    )
    parser.add_argument(
        "--max-ping-hops",
        type=int,
        default=64,
        help="Maximum TTL used while measuring candidate path length.",
    )
    parser.add_argument(
        "--pair-probe-timeout",
        type=int,
        default=2,
        help="Seconds to wait for each candidate ping/TTL response.",
    )
    parser.add_argument(
        "--pair-probe-limit",
        type=int,
        default=1000,
        help="Maximum candidate pairs tested before failing selection.",
    )
    parser.add_argument("--capture-settle-seconds", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.visualization_dir = args.visualization_dir.expanduser().resolve()
    args.topology_dir = args.topology_dir.expanduser().resolve()
    if args.result_dir is not None:
        args.result_dir = args.result_dir.expanduser().resolve()
    if args.ping_groups != 5:
        parser.error("this experiment requires exactly 5 ping groups")
    if args.ping_pair:
        if len(args.ping_pair) != args.ping_groups:
            parser.error("--ping-pair must be repeated exactly five times")
        parsed_pairs = []
        for value in args.ping_pair:
            if "=" not in value:
                parser.error(f"invalid --ping-pair {value!r}; expected SOURCE=DESTINATION")
            source, destination = value.split("=", 1)
            if not source or not destination or source == destination:
                parser.error(f"invalid --ping-pair {value!r}")
            parsed_pairs.append((source, destination))
        args.ping_pair = parsed_pairs
    for name in (
        "compose_jobs",
        "compose_batch_size",
        "verify_jobs",
        "ping_count",
        "min_ping_hops",
        "max_ping_hops",
        "pair_probe_timeout",
        "pair_probe_limit",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_ping_hops < args.min_ping_hops:
        parser.error("--max-ping-hops must be greater than or equal to --min-ping-hops")
    return args


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.result_dir or (
        args.topology_dir
        / "evaluationResult"
        / "traffic_observer"
        / f"{args.output_dir.name}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "experiment.log")
    summary_path = run_dir / "summary.json"
    compose_project = project_name(args.output_dir)
    if summary_path.is_file() and args.start_stage != "build":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("resume_history", []).append(
            {
                "resumed_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "start_stage": args.start_stage,
            }
        )
        summary.pop("error", None)
        summary.pop("finished_at", None)
        summary["success"] = False
        summary["start_stage"] = args.start_stage
        summary["settings"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        summary.setdefault("stages", {})
    else:
        summary = {
            "success": False,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_dir": str(run_dir),
            "output_dir": str(args.output_dir),
            "visualization_dir": str(args.visualization_dir),
            "compose_project": compose_project,
            "start_stage": args.start_stage,
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "stages": {},
        }
    try:
        logger.log(f"experiment started: run_dir={run_dir}")
        summary["visualization_setup"] = prepare_visualization_repo(args, logger)
        summary["preflight"] = check_preflight(args, logger)
        write_json(summary_path, summary)

        if not stage_enabled(args.start_stage, "up") and "up" not in summary["stages"]:
            services = compose_services(
                args.output_dir / "docker-compose.yml", compose_project, logger
            )
            rows = project_container_states(compose_project, logger)
            running_count = sum(row.get("State") == "running" for row in rows)
            graceful_exit_count = sum(
                row.get("State") == "exited"
                and row.get("Status", "").startswith("Exited (0)")
                for row in rows
            )
            present_services = {
                row.get("Service") for row in rows if row.get("Service")
            }
            missing_services = sorted(set(services) - present_services)
            allowed_missing_services = {"seedemu_internet_map"}
            unexpected_missing = sorted(
                set(missing_services) - allowed_missing_services
            )
            if (
                unexpected_missing
                or len(rows) != len(services) - len(missing_services)
                or running_count + graceful_exit_count != len(rows)
            ):
                raise ExperimentError(
                    "cannot resume: existing Compose project is not ready; "
                    f"services={len(services)} created={len(rows)} running={running_count} "
                    f"graceful_exits={graceful_exit_count} "
                    f"missing_services={missing_services} "
                    f"unexpected_missing={unexpected_missing}"
                )
            summary["stages"]["up"] = {
                "success": True,
                "resumed_existing_project": True,
                "project": compose_project,
                "service_count": len(services),
                "created_count": len(rows),
                "running_count": running_count,
                "graceful_exit_count": graceful_exit_count,
                "allowed_missing_services": missing_services,
            }
            write_json(summary_path, summary)

        if (
            not stage_enabled(args.start_stage, "observer")
            and "observer" not in summary["stages"]
        ):
            observer_state = run_capture(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    "seedmu_traffic_observer_service",
                ],
                logger,
            ).stdout.strip()
            filter_status = api_json("GET", "/filter", logger)
            pcap_status = api_json("GET", "/pcap", logger)
            if observer_state != "running":
                raise ExperimentError(
                    f"cannot resume capture: observer state is {observer_state!r}"
                )
            summary["stages"]["observer"] = {
                "success": True,
                "resumed_existing_observer": True,
                "api_url": "http://127.0.0.1:19092",
                "filter": filter_status,
                "pcap": pcap_status,
            }
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "build"):
            logger.log("STAGE build: fast-build and load simulator images")
            summary["stages"]["build"] = build_simulator(args, logger, run_dir)
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "up"):
            logger.log("STAGE up: start simulator containers")
            summary["stages"]["up"] = start_simulator(args, logger, run_dir)
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "start-bird"):
            logger.log("STAGE start-bird: start BIRD and verify protocols")
            summary["stages"]["start_bird"] = start_bird_stage(
                args, logger, run_dir, compose_project
            )
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "kernel"):
            logger.log("STAGE kernel: switch BIRD kernel protocol and verify every AS FIB")
            summary["stages"]["kernel"] = kernel_stage(
                args, logger, run_dir, compose_project
            )
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "observer"):
            logger.log("STAGE observer: build and start traffic observer")
            summary["stages"]["observer"] = observer_stage(args, logger, run_dir)
            write_json(summary_path, summary)

        if stage_enabled(args.start_stage, "capture"):
            logger.log("STAGE capture: run five isolated ICMP capture groups")
            summary["stages"]["capture"] = capture_stage(
                args, logger, run_dir, compose_project
            )
            write_json(summary_path, summary)

        summary["success"] = True
        logger.log("experiment completed successfully")
        returncode = 0
    except Exception as exc:
        summary["error"] = str(exc)
        logger.log(f"EXPERIMENT FAILED: {exc}")
        returncode = 1
    finally:
        summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        write_json(summary_path, summary)
        logger.log(f"summary written: {summary_path}")
        logger.close()
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
