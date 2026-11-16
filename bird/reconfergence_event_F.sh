#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
VM_RUN_ROOT="${VM_RUN_ROOT:-/home/lxl/bird/runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${SCRIPT_DIR}/runs}"

LIMIT_CONTAINERS="${LIMIT_CONTAINERS:-20}"
DOWN_COUNT="${DOWN_COUNT:-1}"
EVENT_MODE="${EVENT_MODE:-selftest}"
WATCH_SECONDS="${WATCH_SECONDS:-45}"
STAGING_TABLE="${STAGING_TABLE:-252}"
CONFIG_WORKERS="${CONFIG_WORKERS:-8}"
WRITE_MAIN="${WRITE_MAIN:-0}"
CLEANUP_STAGING="${CLEANUP_STAGING:-1}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-120}"
LOAD_SAMPLE_INTERVAL="${LOAD_SAMPLE_INTERVAL:-0.2}"

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

This experiment tests an event-style delta source:
  BIRD master4 -> pipe -> t_delta${STAGING_TABLE} -> kernel table ${STAGING_TABLE}
  ip monitor route reads table-${STAGING_TABLE} netlink events
  the agent converts those events to main-FIB ip route batch lines

Environment overrides:
  LIMIT_CONTAINERS=${LIMIT_CONTAINERS}
  EVENT_MODE=${EVENT_MODE}        # selftest or failure
  DOWN_COUNT=${DOWN_COUNT}        # only used by EVENT_MODE=failure
  WATCH_SECONDS=${WATCH_SECONDS}
  STAGING_TABLE=${STAGING_TABLE}
  WRITE_MAIN=${WRITE_MAIN}        # 0 records only, 1 also writes generated batch to main FIB
  CLEANUP_STAGING=${CLEANUP_STAGING}
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
  local run_stamp remote_dir local_dir quoted_project quoted_dir
  run_stamp="$(stamp)"
  remote_dir="${VM_RUN_ROOT}/event-delta-F_${run_stamp}_${PROJECT_NAME}"
  local_dir="${LOCAL_RUN_ROOT}/event-delta-F_${run_stamp}_${PROJECT_NAME}"
  mkdir -p "${local_dir}"
  ssh_vm "mkdir -p '${remote_dir}'"
  quoted_project="$(printf '%q' "${PROJECT_NAME}")"
  quoted_dir="$(printf '%q' "${remote_dir}")"

  set +e
  ssh_vm "sudo -n env PROJECT_NAME=${quoted_project} RUN_DIR=${quoted_dir} LIMIT_CONTAINERS=${LIMIT_CONTAINERS} DOWN_COUNT=${DOWN_COUNT} EVENT_MODE=${EVENT_MODE} WATCH_SECONDS=${WATCH_SECONDS} STAGING_TABLE=${STAGING_TABLE} CONFIG_WORKERS=${CONFIG_WORKERS} WRITE_MAIN=${WRITE_MAIN} CLEANUP_STAGING=${CLEANUP_STAGING} COMMAND_TIMEOUT=${COMMAND_TIMEOUT} LOAD_SAMPLE_INTERVAL=${LOAD_SAMPLE_INTERVAL} python3 -" <<'PY'
import concurrent.futures
import datetime as dt
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import time
import threading
from pathlib import Path

PROJECT = os.environ["PROJECT_NAME"]
RUN_DIR = Path(os.environ["RUN_DIR"])
LIMIT_CONTAINERS = int(os.environ["LIMIT_CONTAINERS"])
DOWN_COUNT = int(os.environ["DOWN_COUNT"])
EVENT_MODE = os.environ["EVENT_MODE"]
WATCH_SECONDS = float(os.environ["WATCH_SECONDS"])
STAGING_TABLE = int(os.environ["STAGING_TABLE"])
CONFIG_WORKERS = int(os.environ["CONFIG_WORKERS"])
WRITE_MAIN = os.environ["WRITE_MAIN"] == "1"
CLEANUP_STAGING = os.environ["CLEANUP_STAGING"] == "1"
COMMAND_TIMEOUT = float(os.environ["COMMAND_TIMEOUT"])
LOAD_SAMPLE_INTERVAL = float(os.environ["LOAD_SAMPLE_INTERVAL"])

ROUTER_RE = re.compile(r"^as\d+brd-r")
AS_RE = re.compile(r"^as(\d+)brd-r")
PREFIX_RE = re.compile(r"^(?:\[[^\]]+\]\s+)?(?:(Deleted)\s+)?(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}|default)\b(.*)$")


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
        return {"samples": len(vals), "load1_min": min(vals), "load1_avg": statistics.fmean(vals), "load1_max": max(vals)}


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(cmd, *, input_text=None, timeout=120, check=True):
    proc = subprocess.run(cmd, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise LabError(f"{' '.join(cmd)}: {msg}")
    return proc


def nsenter_shell(pid, shell_cmd, timeout=COMMAND_TIMEOUT):
    return run_cmd(["nsenter", "-t", str(pid), "-n", "-m", "--", "sh", "-lc", shell_cmd], timeout=timeout, check=False)


def nsenter_input(pid, shell_cmd, input_text, timeout=COMMAND_TIMEOUT):
    return run_cmd(["nsenter", "-t", str(pid), "-n", "-m", "--", "sh", "-lc", shell_cmd], input_text=input_text, timeout=timeout, check=False)


def nsenter_batch(pid, batch, timeout=COMMAND_TIMEOUT):
    return run_cmd(["nsenter", "-t", str(pid), "-n", "-m", "--", "ip", "-force", "-batch", "-"], input_text=batch, timeout=timeout, check=False)


def docker_router_containers(project, limit):
    ps = run_cmd(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}}"])
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        raise LabError(f"no running containers found for project {project}")
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
            continue
        if parts[2].lower() == "true" and pid > 0 and ROUTER_RE.match(name):
            match = AS_RE.match(name)
            routers.append({"name": name, "pid": pid, "as": int(match.group(1)) if match else None})
    routers.sort(key=lambda item: item["name"])
    if limit:
        routers = routers[:limit]
    if not routers:
        raise LabError("no running brd routers found")
    return routers


def staging_config():
    return f"""ipv4 table t_delta{STAGING_TABLE};

protocol pipe delta{STAGING_TABLE}_pipe {{
    table master4;
    peer table t_delta{STAGING_TABLE};
    import none;
    export all;
}}

protocol kernel delta{STAGING_TABLE} {{
    kernel table {STAGING_TABLE};
    merge paths on;
    persist off;
    scan time 3600;
    ipv4 {{
        table t_delta{STAGING_TABLE};
        import none;
        export all;
    }};
}}
"""


def configure_staging(container):
    start = time.monotonic()
    conf_path = f"/etc/bird/conf/delta{STAGING_TABLE}.conf"
    proc = nsenter_input(
        container["pid"],
        f"cat > {conf_path} && birdc configure && ip route flush table {STAGING_TABLE} >/dev/null 2>&1 || true && birdc restart delta{STAGING_TABLE}",
        staging_config(),
    )
    return {
        "name": container["name"],
        "pid": container["pid"],
        "success": proc.returncode == 0,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-2000:],
    }


def cleanup_staging(container):
    start = time.monotonic()
    proc = nsenter_shell(
        container["pid"],
        f"rm -f /etc/bird/conf/delta{STAGING_TABLE}.conf; birdc configure >/dev/null 2>&1 || true; ip route flush table {STAGING_TABLE} >/dev/null 2>&1 || true",
    )
    return {
        "name": container["name"],
        "pid": container["pid"],
        "success": proc.returncode == 0,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-2000:],
    }


def task_summary(results):
    ok = [item["duration_seconds"] for item in results if item.get("success")]
    failed = [item for item in results if not item.get("success")]
    out = {"tasks": len(results), "successes": len(ok), "failures": len(failed), "errors": failed[:20]}
    if ok:
        out.update({"task_duration_min": min(ok), "task_duration_avg": statistics.fmean(ok), "task_duration_max": max(ok)})
    return out


def configure_all(containers):
    start = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, CONFIG_WORKERS)) as pool:
        futures = {pool.submit(configure_staging, item): item for item in containers}
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                item = futures[fut]
                results.append({"name": item["name"], "pid": item["pid"], "success": False, "duration_seconds": 0.0, "error": str(exc)})
    return {"duration_seconds": round(time.monotonic() - start, 6), "task_summary": task_summary(results), "results": results}


def cleanup_all(containers):
    start = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, CONFIG_WORKERS)) as pool:
        futures = {pool.submit(cleanup_staging, item): item for item in containers}
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                item = futures[fut]
                results.append({"name": item["name"], "pid": item["pid"], "success": False, "duration_seconds": 0.0, "error": str(exc)})
    return {"duration_seconds": round(time.monotonic() - start, 6), "task_summary": task_summary(results), "results": results}


def start_monitors(containers):
    monitor_dir = RUN_DIR / "monitors"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitors = []
    for item in containers:
        path = monitor_dir / f"{item['name']}.log"
        fh = path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            ["nsenter", "-t", str(item["pid"]), "-n", "-m", "--", "ip", "-ts", "monitor", "route"],
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitors.append({"container": item, "path": str(path), "fh": fh, "proc": proc})
    return monitors


def stop_monitors(monitors):
    out = []
    for mon in monitors:
        proc = mon["proc"]
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        mon["fh"].close()
        out.append({"name": mon["container"]["name"], "returncode": proc.returncode, "log": mon["path"]})
    return out


def trigger_selftest(containers):
    start = time.monotonic()
    results = []
    for item in containers:
        proc = nsenter_shell(item["pid"], f"birdc disable delta{STAGING_TABLE}_pipe; sleep 2; birdc enable delta{STAGING_TABLE}_pipe", timeout=COMMAND_TIMEOUT)
        results.append(
            {
                "name": item["name"],
                "pid": item["pid"],
                "success": proc.returncode == 0,
                "duration_seconds": 0.0,
                "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-1000:],
            }
        )
    return {"mode": "selftest", "duration_seconds": round(time.monotonic() - start, 6), "task_summary": task_summary(results), "results": results}


def trigger_failure(all_routers, watch_set):
    start = time.monotonic()
    watch_names = {item["name"] for item in watch_set}
    victims = [item for item in all_routers if item["name"] not in watch_names][:DOWN_COUNT]
    results = []
    for item in victims:
        proc = run_cmd(["docker", "stop", "-t", "1", item["name"]], timeout=COMMAND_TIMEOUT, check=False)
        results.append(
            {
                "name": item["name"],
                "as": item.get("as"),
                "success": proc.returncode == 0,
                "duration_seconds": 0.0,
                "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-1000:],
            }
        )
    return {"mode": "failure", "victims": [{"name": item["name"], "as": item.get("as")} for item in victims], "duration_seconds": round(time.monotonic() - start, 6), "task_summary": task_summary(results), "results": results}


def route_text_to_main_batch(prefix, rest, deleted):
    if prefix == "default":
        prefix = "0.0.0.0/0"
    if deleted:
        return f"route del {prefix} proto bird metric 32"
    rest = re.sub(r"\s+table\s+\d+\b", "", rest)
    rest = re.sub(r"\s+", " ", rest).strip()
    return f"route replace {prefix} {rest}".strip()


def parse_monitor_log(path):
    raw_events = 0
    table_events = 0
    prefix_events = 0
    last_by_prefix = {}
    ordered = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw_events += 1
            if f"table {STAGING_TABLE}" not in line:
                continue
            table_events += 1
            text = line.strip()
            match = PREFIX_RE.match(text)
            if not match:
                continue
            prefix = match.group(2)
            deleted = bool(match.group(1))
            rest = match.group(3)
            batch = route_text_to_main_batch(prefix, rest, deleted)
            prefix_events += 1
            last_by_prefix[prefix] = batch
            ordered.append(batch)
    collapsed = [last_by_prefix[prefix] for prefix in sorted(last_by_prefix)]
    return {
        "raw_events": raw_events,
        "table_events": table_events,
        "prefix_events": prefix_events,
        "collapsed_ops": len(collapsed),
        "ordered_ops": len(ordered),
        "batch": "\n".join(collapsed) + ("\n" if collapsed else ""),
    }


def parse_all(monitors):
    start = time.monotonic()
    results = []
    for mon in monitors:
        item = mon["container"]
        parsed = parse_monitor_log(mon["path"])
        batch_path = RUN_DIR / "batches" / f"{item['name']}.batch"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(parsed.pop("batch"), encoding="utf-8")
        parsed.update({"name": item["name"], "pid": item["pid"], "batch_path": str(batch_path), "success": True})
        results.append(parsed)
    return {
        "duration_seconds": round(time.monotonic() - start, 6),
        "containers": len(results),
        "raw_events": sum(item["raw_events"] for item in results),
        "table_events": sum(item["table_events"] for item in results),
        "prefix_events": sum(item["prefix_events"] for item in results),
        "collapsed_ops": sum(item["collapsed_ops"] for item in results),
        "ordered_ops": sum(item["ordered_ops"] for item in results),
        "results": results,
    }


def write_main(containers, parsed):
    by_name = {item["name"]: item for item in containers}
    start = time.monotonic()
    results = []
    for row in parsed["results"]:
        batch = Path(row["batch_path"]).read_text(encoding="utf-8")
        if not batch:
            results.append({"name": row["name"], "success": True, "ops": 0, "duration_seconds": 0.0, "skipped": True})
            continue
        item = by_name[row["name"]]
        t0 = time.monotonic()
        proc = nsenter_batch(item["pid"], batch, timeout=COMMAND_TIMEOUT)
        results.append(
            {
                "name": row["name"],
                "success": proc.returncode == 0,
                "ops": row["collapsed_ops"],
                "duration_seconds": round(time.monotonic() - t0, 6),
                "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-2000:],
            }
        )
    duration = time.monotonic() - start
    ops = sum(item["ops"] for item in results if item["success"])
    return {"duration_seconds": round(duration, 6), "route_ops": ops, "task_summary": task_summary(results), "results": results}


def write_summary(summary):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    started = time.time()
    summary = {
        "schema_version": 1,
        "experiment": "event_delta_F",
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "error": None,
        "host": socket.gethostname(),
        "project": PROJECT,
        "work_dir": str(RUN_DIR),
        "settings": {
            "limit_containers": LIMIT_CONTAINERS,
            "down_count": DOWN_COUNT,
            "event_mode": EVENT_MODE,
            "watch_seconds": WATCH_SECONDS,
            "staging_table": STAGING_TABLE,
            "config_workers": CONFIG_WORKERS,
            "write_main": WRITE_MAIN,
            "cleanup_staging": CLEANUP_STAGING,
        },
        "stages": {},
    }
    write_summary(summary)
    monitors = []
    containers = []
    try:
        all_routers = docker_router_containers(PROJECT, 0)
        containers = all_routers[:LIMIT_CONTAINERS] if LIMIT_CONTAINERS else all_routers
        summary["running_brdnodes_total"] = len(all_routers)
        summary["watch_containers"] = [{"name": item["name"], "pid": item["pid"], "as": item.get("as")} for item in containers]
        write_summary(summary)

        configured = configure_all(containers)
        summary["stages"]["configure_staging"] = configured
        write_summary(summary)
        if configured["task_summary"]["failures"]:
            raise LabError("failed to configure staging table in some watchers")

        monitors = start_monitors(containers)
        time.sleep(3)
        with LoadMonitor(LOAD_SAMPLE_INTERVAL) as monitor:
            if EVENT_MODE == "selftest":
                trigger = trigger_selftest(containers)
            elif EVENT_MODE == "failure":
                trigger = trigger_failure(all_routers, containers)
            else:
                raise LabError(f"unsupported EVENT_MODE={EVENT_MODE!r}")
            summary["stages"]["trigger"] = trigger
            write_summary(summary)
            remaining = max(0.0, WATCH_SECONDS - trigger["duration_seconds"])
            if remaining:
                time.sleep(remaining)
        summary["stages"]["watch_load"] = monitor.stats()

        monitor_stop = stop_monitors(monitors)
        monitors = []
        summary["stages"]["monitor_stop"] = monitor_stop
        parsed = parse_all([{"container": item, "path": str(RUN_DIR / "monitors" / f"{item['name']}.log")} for item in containers])
        summary["stages"]["parse_events"] = parsed
        write_summary(summary)

        if WRITE_MAIN:
            summary["stages"]["write_main"] = write_main(containers, parsed)
            write_summary(summary)

        if CLEANUP_STAGING:
            summary["stages"]["cleanup_staging"] = cleanup_all(containers)
            write_summary(summary)

        summary["success"] = parsed["prefix_events"] > 0 and (not WRITE_MAIN or summary["stages"]["write_main"]["task_summary"]["failures"] == 0)
        if not summary["success"]:
            summary["error"] = "no prefix-level staging route events were captured"
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", flush=True)
    finally:
        if monitors:
            stop_monitors(monitors)
        summary["ended_at"] = now_iso()
        summary["duration_seconds"] = round(time.time() - started, 6)
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
    "running_brdnodes_total": s.get("running_brdnodes_total"),
    "watchers": len(s.get("watch_containers", [])),
    "settings": s.get("settings"),
    "configure": st.get("configure_staging", {}).get("task_summary"),
    "trigger": {k: st.get("trigger", {}).get(k) for k in ["mode", "duration_seconds", "victims"]},
    "parse_events": {k: st.get("parse_events", {}).get(k) for k in ["raw_events", "table_events", "prefix_events", "ordered_ops", "collapsed_ops", "duration_seconds"]},
    "write_main": {k: st.get("write_main", {}).get(k) for k in ["duration_seconds", "route_ops", "task_summary"]},
    "cleanup": st.get("cleanup_staging", {}).get("task_summary"),
    "duration_seconds": s.get("duration_seconds"),
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
