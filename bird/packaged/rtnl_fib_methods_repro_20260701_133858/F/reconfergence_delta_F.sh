#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
VM_RUN_ROOT="${VM_RUN_ROOT:-/home/lxl/bird/runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${SCRIPT_DIR}/runs}"

DOWN_COUNT="${DOWN_COUNT:-10}"
STOP_TIMEOUT="${STOP_TIMEOUT:-1}"
SELECT_STRATEGY="${SELECT_STRATEGY:-spread}"
POST_DOWN_SLEEP_SECONDS="${POST_DOWN_SLEEP_SECONDS:-30}"
LIMIT_CONTAINERS="${LIMIT_CONTAINERS:-0}"

DUMP_WORKERS="${DUMP_WORKERS:-32}"
FIB_DUMP_WORKERS="${FIB_DUMP_WORKERS:-${DUMP_WORKERS}}"
F_SHARDS="${F_SHARDS:-8}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-180}"
SAMPLE_COUNT="${SAMPLE_COUNT:-10}"
FIB_RIB_MIN_DELTA="${FIB_RIB_MIN_DELTA:-0}"
LOAD_SAMPLE_INTERVAL="${LOAD_SAMPLE_INTERVAL:-0.2}"
FIB_AWARE_DELETE="${FIB_AWARE_DELETE:-1}"
WRITE_MODE="${WRITE_MODE:-setns-ip}"
REMOTE_ROOT_PYTHON="${REMOTE_ROOT_PYTHON:-1}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

usage() {
  cat <<EOF
Usage:
  $0 [run]
  $0 status

Environment overrides:
  VM_HOST=${VM_HOST}
  VM_USER=${VM_USER}
  PROJECT_NAME=${PROJECT_NAME}
  DOWN_COUNT=${DOWN_COUNT}
  POST_DOWN_SLEEP_SECONDS=${POST_DOWN_SLEEP_SECONDS}
  DUMP_WORKERS=${DUMP_WORKERS}
  FIB_DUMP_WORKERS=${FIB_DUMP_WORKERS}
  F_SHARDS=${F_SHARDS}
  FIB_AWARE_DELETE=${FIB_AWARE_DELETE}
  WRITE_MODE=${WRITE_MODE}
  REMOTE_ROOT_PYTHON=${REMOTE_ROOT_PYTHON}

The run phase snapshots current BIRD RIB, stops DOWN_COUNT brdnode containers
from different ASes, snapshots the new BIRD RIB, diffs the two snapshots, and
writes only added/changed/deleted routes to main FIB.
EOF
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

rsync_from_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "${VM_USER}@${VM_HOST}:$1" "$2"
}

stamp() {
  date +%Y%m%d_%H%M%S
}

phase_status() {
  ssh_vm 'date -Is; cat /proc/loadavg; printf "containers="; docker ps -aq | wc -l; printf "running="; docker ps -q | wc -l; printf "project_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' -q | wc -l; printf "brd_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' --format "{{.Names}}" | grep -Ec "^as[0-9]+brd-r" || true; printf "bird="; pgrep -fc "(^|/)bird( |$)" || true'
}

phase_run() {
  local run_stamp remote_dir local_dir
  run_stamp="$(stamp)"
  remote_dir="${VM_RUN_ROOT}/reconfergence-delta-F_${run_stamp}_${PROJECT_NAME}"
  local_dir="${LOCAL_RUN_ROOT}/reconfergence-delta-F_${run_stamp}_${PROJECT_NAME}"
  mkdir -p "${local_dir}"
  ssh_vm "mkdir -p '${remote_dir}'"

  local quoted_project quoted_dir
  quoted_project="$(printf '%q' "${PROJECT_NAME}")"
  quoted_dir="$(printf '%q' "${remote_dir}")"

  set +e
  local remote_python_prefix
  if [ "${REMOTE_ROOT_PYTHON}" = "1" ]; then
    remote_python_prefix="sudo -n env"
  else
    remote_python_prefix="env"
  fi

  ssh_vm "${remote_python_prefix} PROJECT_NAME=${quoted_project} RUN_DIR=${quoted_dir} DOWN_COUNT=${DOWN_COUNT} STOP_TIMEOUT=${STOP_TIMEOUT} SELECT_STRATEGY=${SELECT_STRATEGY} POST_DOWN_SLEEP_SECONDS=${POST_DOWN_SLEEP_SECONDS} LIMIT_CONTAINERS=${LIMIT_CONTAINERS} DUMP_WORKERS=${DUMP_WORKERS} FIB_DUMP_WORKERS=${FIB_DUMP_WORKERS} F_SHARDS=${F_SHARDS} COMMAND_TIMEOUT=${COMMAND_TIMEOUT} SAMPLE_COUNT=${SAMPLE_COUNT} FIB_RIB_MIN_DELTA=${FIB_RIB_MIN_DELTA} LOAD_SAMPLE_INTERVAL=${LOAD_SAMPLE_INTERVAL} FIB_AWARE_DELETE=${FIB_AWARE_DELETE} WRITE_MODE=${WRITE_MODE} python3 -" <<'PY'
import concurrent.futures
import ctypes
import datetime as dt
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import time
import threading
import multiprocessing
from pathlib import Path

PROJECT = os.environ["PROJECT_NAME"]
RUN_DIR = Path(os.environ["RUN_DIR"])
DOWN_COUNT = int(os.environ["DOWN_COUNT"])
STOP_TIMEOUT = int(os.environ["STOP_TIMEOUT"])
SELECT_STRATEGY = os.environ["SELECT_STRATEGY"]
POST_DOWN_SLEEP_SECONDS = float(os.environ["POST_DOWN_SLEEP_SECONDS"])
LIMIT_CONTAINERS = int(os.environ["LIMIT_CONTAINERS"])
DUMP_WORKERS = int(os.environ["DUMP_WORKERS"])
FIB_DUMP_WORKERS = int(os.environ["FIB_DUMP_WORKERS"])
F_SHARDS = int(os.environ["F_SHARDS"])
COMMAND_TIMEOUT = float(os.environ["COMMAND_TIMEOUT"])
SAMPLE_COUNT = int(os.environ["SAMPLE_COUNT"])
FIB_RIB_MIN_DELTA = int(os.environ["FIB_RIB_MIN_DELTA"])
LOAD_SAMPLE_INTERVAL = float(os.environ["LOAD_SAMPLE_INTERVAL"])
FIB_AWARE_DELETE = os.environ["FIB_AWARE_DELETE"] == "1"
WRITE_MODE = os.environ["WRITE_MODE"]

ROUTER_RE = re.compile(r"^as\d+brd-r")
AS_RE = re.compile(r"^as(\d+)brd-r")
IPV4_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\s+unicast\b.*\*")
VIA_RE = re.compile(r"^\s+via\s+(\d{1,3}(?:\.\d{1,3}){3})\s+on\s+(\S+)")
DEV_RE = re.compile(r"^\s+dev\s+(\S+)")
FIB_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}|default)\b")
CLONE_NEWNET = 0x40000000
LIBC = ctypes.CDLL(None, use_errno=True)


class LabError(RuntimeError):
    pass


class LoadMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                load1, load5, load15 = os.getloadavg()
                self.samples.append({"t": time.time(), "load1": load1, "load5": load5, "load15": load15})
            except OSError:
                pass
            self._stop.wait(self.interval)

    def stats(self):
        if not self.samples:
            return {"samples": 0}
        vals = [item["load1"] for item in self.samples]
        return {
            "samples": len(vals),
            "load1_min": min(vals),
            "load1_avg": statistics.fmean(vals),
            "load1_max": max(vals),
        }


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(cmd, *, input_text=None, timeout=120, check=True):
    proc = subprocess.run(cmd, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise LabError(f"{' '.join(cmd)}: {msg}")
    return proc


def nsenter_prefix():
    if os.geteuid() == 0:
        return ["nsenter"]
    if shutil.which("sudo"):
        check = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            return ["sudo", "-n", "nsenter"]
    raise LabError("root or passwordless sudo is required for nsenter")


NSENTER = nsenter_prefix()


def nsenter_shell(pid, shell_cmd, timeout):
    cmd = NSENTER + ["-t", str(pid), "-n", "-m", "--", "sh", "-lc", shell_cmd]
    return run_cmd(cmd, timeout=timeout, check=False)


def nsenter_batch(pid, batch, timeout):
    cmd = NSENTER + ["-t", str(pid), "-n", "-m", "--", "ip", "-force", "-batch", "-"]
    return run_cmd(cmd, input_text=batch, timeout=timeout, check=False)


def setns_net(pid):
    ns_path = f"/proc/{pid}/ns/net"
    fd = os.open(ns_path, os.O_RDONLY)
    try:
        if hasattr(os, "setns"):
            os.setns(fd, CLONE_NEWNET)
        else:
            rc = LIBC.setns(fd, CLONE_NEWNET)
            if rc != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), ns_path)
    finally:
        os.close(fd)


def setns_ip_batch(pid, batch, timeout):
    setns_net(pid)
    return run_cmd(["ip", "-force", "-batch", "-"], input_text=batch, timeout=timeout, check=False)


def task_summary(results):
    ok = [item["duration_seconds"] for item in results if item["success"]]
    failed = [item for item in results if not item["success"]]
    out = {
        "tasks": len(results),
        "successes": len(ok),
        "failures": len(failed),
        "errors": failed[:20],
    }
    if ok:
        out.update(
            {
                "task_duration_min": min(ok),
                "task_duration_avg": statistics.fmean(ok),
                "task_duration_max": max(ok),
            }
        )
    return out


def parse_network_count(text):
    matches = re.findall(r"(\d+)\s+networks?", text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"network[s]?\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def parse_bird_primary_routes(text):
    routes = []
    current = None
    for line in text.splitlines():
        route_match = IPV4_PREFIX_RE.match(line)
        if route_match:
            if current and current["nexthops"]:
                routes.append(current)
            current = {"prefix": route_match.group(1), "nexthops": []}
            continue
        if current is None:
            continue
        via_match = VIA_RE.match(line)
        if via_match:
            current["nexthops"].append({"via": via_match.group(1), "dev": via_match.group(2)})
            continue
        dev_match = DEV_RE.match(line)
        if dev_match:
            current["nexthops"].append({"dev": dev_match.group(1)})
    if current and current["nexthops"]:
        routes.append(current)
    return routes


def route_to_ip_batch_line(route):
    prefix = route["prefix"]
    nexthops = route["nexthops"]
    if not nexthops:
        return None
    if len(nexthops) == 1:
        hop = nexthops[0]
        if hop.get("via"):
            return f"route replace {prefix} via {hop['via']} dev {hop['dev']} proto bird metric 32"
        return f"route replace {prefix} dev {hop['dev']} proto bird metric 32"
    parts = [f"route replace {prefix} proto bird metric 32"]
    for hop in nexthops:
        if hop.get("via"):
            parts.append(f"nexthop via {hop['via']} dev {hop['dev']}")
        else:
            parts.append(f"nexthop dev {hop['dev']}")
    return " ".join(parts)


def route_delete_line(prefix):
    return f"route del {prefix} proto bird metric 32"


def docker_router_containers(project, limit):
    ps = run_cmd(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ]
    )
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        raise LabError(f"no running containers found for compose project {project}")
    inspect = run_cmd(["docker", "inspect", "-f", "{{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids)
    routers = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0].lstrip("/")
        try:
            pid = int(parts[1])
        except ValueError:
            pid = 0
        if parts[2].lower() == "true" and pid > 0 and ROUTER_RE.match(name):
            as_match = AS_RE.match(name)
            routers.append({"name": name, "pid": pid, "as": int(as_match.group(1)) if as_match else None})
    routers.sort(key=lambda item: item["name"])
    if limit:
        routers = routers[:limit]
    if not routers:
        raise LabError(f"no running router containers matched project {project}")
    return routers


def choose_victims(containers, count):
    groups = {}
    for item in containers:
        asn = item.get("as")
        if asn is None:
            continue
        groups.setdefault(asn, []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: item["name"])
    asns = sorted(groups)
    if count <= 0:
        return []
    if count > len(asns):
        raise LabError(f"requested {count} ASes but only {len(asns)} running AS groups exist")
    if SELECT_STRATEGY == "first":
        chosen = asns[:count]
    elif SELECT_STRATEGY == "spread":
        if count == 1:
            chosen = [asns[0]]
        else:
            chosen = []
            used = set()
            for idx in range(count):
                pos = round(idx * (len(asns) - 1) / (count - 1))
                while pos in used and pos + 1 < len(asns):
                    pos += 1
                while pos in used and pos - 1 >= 0:
                    pos -= 1
                used.add(pos)
                chosen.append(asns[pos])
    else:
        raise LabError(f"unsupported SELECT_STRATEGY={SELECT_STRATEGY!r}")
    return [groups[asn][0] for asn in chosen]


def dump_one(container):
    start = time.monotonic()
    proc = nsenter_shell(container["pid"], "birdc show route primary", COMMAND_TIMEOUT)
    duration = time.monotonic() - start
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {
            "name": container["name"],
            "pid": container["pid"],
            "as": container.get("as"),
            "success": False,
            "duration_seconds": round(duration, 6),
            "error": text[-2000:],
            "routes": {},
        }
    routes = {}
    for route in parse_bird_primary_routes(text):
        line = route_to_ip_batch_line(route)
        if line:
            routes[route["prefix"]] = line
    return {
        "name": container["name"],
        "pid": container["pid"],
        "as": container.get("as"),
        "success": True,
        "duration_seconds": round(duration, 6),
        "route_count": len(routes),
        "routes": routes,
    }


def dump_all(label, containers):
    start = time.monotonic()
    results = []
    route_maps = {}
    with LoadMonitor(LOAD_SAMPLE_INTERVAL) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, DUMP_WORKERS)) as pool:
            futures = {pool.submit(dump_one, item): item for item in containers}
            for fut in concurrent.futures.as_completed(futures):
                item = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "name": item["name"],
                        "pid": item["pid"],
                        "as": item.get("as"),
                        "success": False,
                        "duration_seconds": 0.0,
                        "error": str(exc),
                        "routes": {},
                    }
                if row["success"]:
                    route_maps[row["name"]] = row.pop("routes")
                else:
                    row.pop("routes", None)
                results.append(row)
    failures = [row for row in results if not row["success"]]
    total_routes = sum(row.get("route_count", 0) for row in results if row["success"])
    return route_maps, {
        "label": label,
        "containers": len(containers),
        "duration_seconds": round(time.monotonic() - start, 6),
        "total_routes": total_routes,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
        "errors": failures[:20],
    }


def parse_fib_prefixes(text):
    prefixes = set()
    for line in text.splitlines():
        match = FIB_PREFIX_RE.match(line.strip())
        if not match:
            continue
        prefix = match.group(1)
        if prefix == "default":
            prefix = "0.0.0.0/0"
        prefixes.add(prefix)
    return prefixes


def dump_fib_one(container):
    start = time.monotonic()
    proc = nsenter_shell(container["pid"], "ip -o route show proto bird", COMMAND_TIMEOUT)
    duration = time.monotonic() - start
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {
            "name": container["name"],
            "pid": container["pid"],
            "as": container.get("as"),
            "success": False,
            "duration_seconds": round(duration, 6),
            "error": text[-2000:],
            "prefixes": set(),
        }
    prefixes = parse_fib_prefixes(text)
    return {
        "name": container["name"],
        "pid": container["pid"],
        "as": container.get("as"),
        "success": True,
        "duration_seconds": round(duration, 6),
        "fib_route_count": len(prefixes),
        "prefixes": prefixes,
    }


def dump_fib_all(containers):
    start = time.monotonic()
    results = []
    fib_maps = {}
    workers = max(1, FIB_DUMP_WORKERS)
    with LoadMonitor(LOAD_SAMPLE_INTERVAL) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(dump_fib_one, item): item for item in containers}
            for fut in concurrent.futures.as_completed(futures):
                item = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "name": item["name"],
                        "pid": item["pid"],
                        "as": item.get("as"),
                        "success": False,
                        "duration_seconds": 0.0,
                        "error": str(exc),
                        "prefixes": set(),
                    }
                if row["success"]:
                    fib_maps[row["name"]] = row.pop("prefixes")
                else:
                    row.pop("prefixes", None)
                results.append(row)
    failures = [row for row in results if not row["success"]]
    total_routes = sum(row.get("fib_route_count", 0) for row in results if row["success"])
    return fib_maps, {
        "label": "current_fib_before_delta_write",
        "containers": len(containers),
        "workers": workers,
        "duration_seconds": round(time.monotonic() - start, 6),
        "total_routes": total_routes,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
        "errors": failures[:20],
    }


def stop_victims(victims):
    out = []
    start_epoch = time.time()
    for victim in victims:
        start = time.monotonic()
        proc = run_cmd(["docker", "stop", "-t", str(STOP_TIMEOUT), victim["name"]], timeout=COMMAND_TIMEOUT, check=False)
        out.append(
            {
                "as": victim.get("as"),
                "name": victim["name"],
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "duration_seconds": round(time.monotonic() - start, 6),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    return {
        "started_epoch": start_epoch,
        "completed_epoch": time.time(),
        "victims": [{"as": item.get("as"), "name": item["name"]} for item in victims],
        "results": out,
        "success": all(item["success"] for item in out),
    }


def diff_one(container, before_routes, after_routes, fib_routes=None):
    before = before_routes.get(container["name"], {})
    after = after_routes.get(container["name"], {})
    fib_prefixes = fib_routes.get(container["name"], set()) if fib_routes is not None else None
    added = []
    changed = []
    unchanged = 0
    for prefix, line in after.items():
        old = before.get(prefix)
        if old is None:
            added.append(line)
        elif old != line:
            changed.append(line)
        else:
            unchanged += 1
    delete_candidates = set(before) - set(after)
    delete_filtered = 0
    if fib_prefixes is not None:
        before_filter = len(delete_candidates)
        delete_candidates &= fib_prefixes
        delete_filtered = before_filter - len(delete_candidates)
    deleted = [route_delete_line(prefix) for prefix in sorted(delete_candidates)]
    batch_lines = added + changed + deleted
    return {
        "name": container["name"],
        "pid": container["pid"],
        "as": container.get("as"),
        "added": len(added),
        "changed": len(changed),
        "deleted": len(deleted),
        "delete_filtered": delete_filtered,
        "unchanged": unchanged,
        "ops": len(batch_lines),
        "batch": "\n".join(batch_lines) + ("\n" if batch_lines else ""),
    }


def write_one_delta(item):
    start = time.monotonic()
    if not item["batch"]:
        return {
            "name": item["name"],
            "pid": item["pid"],
            "as": item.get("as"),
            "success": True,
            "duration_seconds": 0.0,
            "ops": 0,
            "added": item["added"],
            "changed": item["changed"],
            "deleted": item["deleted"],
            "delete_filtered": item.get("delete_filtered", 0),
            "unchanged": item["unchanged"],
            "skipped": True,
            "error": "",
        }
    if WRITE_MODE == "setns-ip" and os.geteuid() == 0:
        proc = setns_ip_batch(item["pid"], item["batch"], COMMAND_TIMEOUT)
        effective_write_mode = "setns-ip"
    else:
        proc = nsenter_batch(item["pid"], item["batch"], COMMAND_TIMEOUT)
        effective_write_mode = "nsenter-ip"
    error_text = proc.stderr.strip() or proc.stdout.strip()
    benign_delete_miss = False
    if proc.returncode != 0 and error_text:
        meaningful = [line.strip() for line in error_text.splitlines() if line.strip()]
        benign_delete_miss = bool(meaningful) and all(
            line.startswith("RTNETLINK answers: No such process") or line.startswith("Command failed -:")
            for line in meaningful
        )
    return {
        "name": item["name"],
        "pid": item["pid"],
        "as": item.get("as"),
        "success": proc.returncode == 0 or benign_delete_miss,
        "duration_seconds": round(time.monotonic() - start, 6),
        "ops": item["ops"],
        "added": item["added"],
        "changed": item["changed"],
        "deleted": item["deleted"],
        "delete_filtered": item.get("delete_filtered", 0),
        "unchanged": item["unchanged"],
        "skipped": False,
        "write_mode": effective_write_mode,
        "benign_delete_miss": benign_delete_miss,
        "returncode": proc.returncode,
        "error": "" if proc.returncode == 0 or benign_delete_miss else error_text[-2000:],
    }


def write_delta(delta_items):
    start = time.monotonic()
    results = []
    write_items = [item for item in delta_items if item["ops"] > 0]
    skipped = [item for item in delta_items if item["ops"] == 0]
    use_setns_pool = WRITE_MODE == "setns-ip" and os.geteuid() == 0
    with LoadMonitor(LOAD_SAMPLE_INTERVAL) as monitor:
        executor = concurrent.futures.ProcessPoolExecutor if use_setns_pool else concurrent.futures.ThreadPoolExecutor
        executor_kwargs = {"max_workers": max(1, F_SHARDS)}
        if use_setns_pool:
            executor_kwargs["mp_context"] = multiprocessing.get_context("fork")
        with executor(**executor_kwargs) as pool:
            futures = {pool.submit(write_one_delta, item): item for item in write_items}
            for fut in concurrent.futures.as_completed(futures):
                item = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append(
                        {
                            "name": item["name"],
                            "pid": item["pid"],
                            "as": item.get("as"),
                            "success": False,
                            "duration_seconds": 0.0,
                            "ops": item["ops"],
                            "added": item["added"],
                            "changed": item["changed"],
                            "deleted": item["deleted"],
                            "delete_filtered": item.get("delete_filtered", 0),
                            "unchanged": item["unchanged"],
                            "skipped": False,
                            "error": str(exc),
                        }
                    )
    for item in skipped:
        results.append(write_one_delta(item))
    duration = time.monotonic() - start
    ops = sum(item["ops"] for item in results if item["success"])
    return {
        "workers": F_SHARDS,
        "write_mode": "setns-ip" if use_setns_pool else "nsenter-ip",
        "containers": len(delta_items),
        "containers_with_delta": sum(1 for item in delta_items if item["ops"] > 0),
        "route_ops": ops,
        "added": sum(item.get("added", 0) for item in results if item["success"]),
        "changed": sum(item.get("changed", 0) for item in results if item["success"]),
        "deleted": sum(item.get("deleted", 0) for item in results if item["success"]),
        "delete_filtered": sum(item.get("delete_filtered", 0) for item in results if item["success"]),
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(ops / duration, 3) if duration > 0 else None,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
        "results": results,
    }


def sample_different_as(containers, count):
    selected = []
    seen = set()
    for item in sorted(containers, key=lambda row: (row.get("as") is None, row.get("as") or 0, row["name"])):
        asn = item.get("as")
        if asn is not None and asn in seen:
            continue
        selected.append(item)
        if asn is not None:
            seen.add(asn)
        if len(selected) >= count:
            return selected
    names = {item["name"] for item in selected}
    for item in sorted(containers, key=lambda row: row["name"]):
        if item["name"] in names:
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def verify_samples(containers):
    samples = []
    for item in sample_different_as(containers, SAMPLE_COUNT):
        proc = nsenter_shell(item["pid"], "printf 'IP_ROUTE_LINES='; ip route | wc -l; birdc show route count", COMMAND_TIMEOUT)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ip_match = re.search(r"IP_ROUTE_LINES=(\d+)", text)
        ip_routes = int(ip_match.group(1)) if ip_match else None
        networks = parse_network_count(text)
        passed = (
            proc.returncode == 0
            and ip_routes is not None
            and networks is not None
            and ip_routes >= networks + FIB_RIB_MIN_DELTA
        )
        samples.append(
            {
                "name": item["name"],
                "pid": item["pid"],
                "as": item.get("as"),
                "success": passed,
                "ip_route_lines": ip_routes,
                "bird_networks": networks,
                "returncode": proc.returncode,
                "raw_tail": text[-1000:],
            }
        )
    return {
        "sample_count": len(samples),
        "successes": sum(1 for row in samples if row["success"]),
        "failures": sum(1 for row in samples if not row["success"]),
        "samples": samples,
        "success": bool(samples) and all(row["success"] for row in samples),
        "condition": f"ip route | wc -l >= birdc show route count networks + {FIB_RIB_MIN_DELTA}",
    }


def write_summary(summary):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    started_epoch = time.time()
    summary = {
        "schema_version": 1,
        "experiment": "reconfergence_delta_F",
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "error": None,
        "host": socket.gethostname(),
        "project": PROJECT,
        "work_dir": str(RUN_DIR),
        "settings": {
            "down_count": DOWN_COUNT,
            "stop_timeout": STOP_TIMEOUT,
            "select_strategy": SELECT_STRATEGY,
            "post_down_sleep_seconds": POST_DOWN_SLEEP_SECONDS,
            "limit_containers": LIMIT_CONTAINERS,
            "dump_workers": DUMP_WORKERS,
            "fib_dump_workers": FIB_DUMP_WORKERS,
            "f_shards": F_SHARDS,
            "command_timeout": COMMAND_TIMEOUT,
            "sample_count": SAMPLE_COUNT,
            "fib_rib_min_delta": FIB_RIB_MIN_DELTA,
            "fib_aware_delete": FIB_AWARE_DELETE,
            "write_mode": WRITE_MODE,
            "remote_euid": os.geteuid(),
        },
        "stages": {},
    }
    write_summary(summary)
    try:
        containers_before = docker_router_containers(PROJECT, LIMIT_CONTAINERS)
        summary["running_brdnodes_before"] = len(containers_before)
        write_summary(summary)

        before_routes, baseline = dump_all("baseline_before_failure", containers_before)
        summary["stages"]["baseline_dump"] = baseline
        write_summary(summary)
        if baseline["task_summary"]["failures"]:
            raise LabError("baseline BIRD RIB dump failed")

        victims = choose_victims(containers_before, DOWN_COUNT)
        event = stop_victims(victims)
        summary["stages"]["failure_event"] = event
        write_summary(summary)
        if not event["success"]:
            raise LabError("failed to stop all selected brdnode containers")

        if POST_DOWN_SLEEP_SECONDS > 0:
            time.sleep(POST_DOWN_SLEEP_SECONDS)

        containers_after = docker_router_containers(PROJECT, LIMIT_CONTAINERS)
        summary["running_brdnodes_after"] = len(containers_after)
        after_routes, current = dump_all("current_after_failure", containers_after)
        summary["stages"]["current_dump"] = current
        write_summary(summary)
        if current["task_summary"]["failures"]:
            raise LabError("current BIRD RIB dump failed")

        fib_routes = None
        if FIB_AWARE_DELETE:
            fib_routes, fib_dump = dump_fib_all(containers_after)
            summary["stages"]["fib_dump"] = fib_dump
            write_summary(summary)
            if fib_dump["task_summary"]["failures"]:
                raise LabError("current FIB dump failed")

        diff_started = time.monotonic()
        delta_items = [diff_one(item, before_routes, after_routes, fib_routes) for item in containers_after]
        diff_duration = time.monotonic() - diff_started
        diff_summary = {
            "duration_seconds": round(diff_duration, 6),
            "containers": len(delta_items),
            "containers_with_delta": sum(1 for item in delta_items if item["ops"] > 0),
            "added": sum(item["added"] for item in delta_items),
            "changed": sum(item["changed"] for item in delta_items),
            "deleted": sum(item["deleted"] for item in delta_items),
            "delete_filtered": sum(item.get("delete_filtered", 0) for item in delta_items),
            "unchanged": sum(item["unchanged"] for item in delta_items),
            "route_ops": sum(item["ops"] for item in delta_items),
            "top_delta": sorted(
                [
                    {
                        "name": item["name"],
                        "as": item.get("as"),
                        "ops": item["ops"],
                        "added": item["added"],
                        "changed": item["changed"],
                        "deleted": item["deleted"],
                        "delete_filtered": item.get("delete_filtered", 0),
                    }
                    for item in delta_items
                ],
                key=lambda row: row["ops"],
                reverse=True,
            )[:20],
        }
        summary["stages"]["diff"] = diff_summary
        write_summary(summary)

        write = write_delta(delta_items)
        summary["stages"]["delta_write"] = write
        write_summary(summary)
        if write["task_summary"]["failures"]:
            raise LabError("delta FIB write failed")

        verify = verify_samples(containers_after)
        summary["stages"]["fib_rib_sample"] = verify
        if not verify["success"]:
            raise LabError("FIB/RIB sample validation failed")

        stop_completed = event.get("completed_epoch")
        summary["timing"] = {
            "total_wall_seconds": round(time.time() - started_epoch, 6),
            "failure_event_to_done_seconds": round(time.time() - stop_completed, 6) if stop_completed else None,
            "post_fault_manual_refresh_seconds": round(
                POST_DOWN_SLEEP_SECONDS
                + current["duration_seconds"]
                + summary["stages"].get("fib_dump", {}).get("duration_seconds", 0)
                + diff_summary["duration_seconds"]
                + write["duration_seconds"],
                6,
            ),
        }
        summary["success"] = True
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", flush=True)
    finally:
        summary["ended_at"] = now_iso()
        write_summary(summary)

    print(f"Summary: {RUN_DIR / 'summary.json'}", flush=True)
    return 0 if summary["success"] else 1


raise SystemExit(main())
PY
  local rc=$?
  set -e

  rsync_from_vm "${remote_dir}/" "${local_dir}/" || true
  if [ -f "${local_dir}/summary.json" ]; then
    python3 - "${local_dir}/summary.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1]))
st = s.get("stages", {})
out = {
    "success": s.get("success"),
    "error": s.get("error"),
    "running_brdnodes_before": s.get("running_brdnodes_before"),
    "running_brdnodes_after": s.get("running_brdnodes_after"),
    "victims": st.get("failure_event", {}).get("victims", []),
    "baseline_dump_seconds": st.get("baseline_dump", {}).get("duration_seconds"),
    "current_dump_seconds": st.get("current_dump", {}).get("duration_seconds"),
    "fib_dump_seconds": st.get("fib_dump", {}).get("duration_seconds"),
    "diff": {k: st.get("diff", {}).get(k) for k in ["duration_seconds", "containers_with_delta", "added", "changed", "deleted", "delete_filtered", "unchanged", "route_ops"]},
    "delta_write": {k: st.get("delta_write", {}).get(k) for k in ["duration_seconds", "write_mode", "route_ops", "route_ops_per_second", "containers_with_delta", "delete_filtered"]},
    "fib_rib_sample": {k: st.get("fib_rib_sample", {}).get(k) for k in ["success", "sample_count", "successes", "failures"]},
    "timing": s.get("timing"),
}
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
  fi
  echo "LOCAL_DIR=${local_dir}"
  echo "REMOTE_DIR=${remote_dir}"
  return "${rc}"
}

main() {
  local phase="${1:-run}"
  case "${phase}" in
    run) phase_run ;;
    status) phase_status ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
