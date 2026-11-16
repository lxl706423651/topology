#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RISK_PATTERN = (
    "BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|"
    "general protection|unregister_netdevice|fib_create_info|bird_rtbulk|"
    "ASSERT_RTNL|soft lockup|hung task"
)


def sh(cmd, *, check=True, text=True, timeout=None):
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        executable="/bin/bash",
    )


def kernel_release():
    return sh("uname -r").stdout.strip()


def percentile(values, pct):
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (pct / 100.0) * (len(data) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (rank - lo)


def summary_stats(values):
    return {
        "count": len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def write_tsv(path, rows, fields):
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in fields) + "\n")


def standard_route_bench(rep, n, out_dir):
    ns = f"rtbulk-sideeffect-{os.getpid()}-{rep}"
    script = rf'''
set -euo pipefail
NS="{ns}"
cleanup() {{
  ip netns del "$NS" 2>/dev/null || true
}}
trap cleanup EXIT

ip netns add "$NS"
ip -n "$NS" link set lo up
ip -n "$NS" link add dummy0 type dummy
ip -n "$NS" link set dummy0 up

start=$(date +%s.%N)
for i in $(seq 1 "{n}"); do
  a=$(( (i / 65536) % 256 ))
  b=$(( (i / 256) % 256 ))
  c=$(( i % 256 ))
  ip -n "$NS" route replace "10.${{a}}.${{b}}.${{c}}/32" dev dummy0 proto static metric 100
done
end=$(date +%s.%N)
routes=$(ip -n "$NS" route show | wc -l)
python3 - <<PY
print("add\t{n}\t%.6f\t$routes" % ($end - $start))
PY

start=$(date +%s.%N)
ip -n "$NS" route flush proto static >/dev/null
end=$(date +%s.%N)
routes=$(ip -n "$NS" route show | wc -l)
python3 - <<PY
print("flush\t{n}\t%.6f\t$routes" % ($end - $start))
PY
'''
    proc = sh(script, timeout=max(600, n // 25))
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[0] in {"add", "flush"}:
            rows.append(
                {
                    "rep": rep,
                    "phase": parts[0],
                    "n": int(parts[1]),
                    "seconds": float(parts[2]),
                    "routes": int(parts[3]),
                }
            )
    if len(rows) != 2:
        raise RuntimeError(f"standard route bench parse failed: {proc.stdout!r} {proc.stderr!r}")
    return rows


def forwarding_bench(rep, seconds):
    suffix = f"{os.getpid()}-{rep}"
    use_iperf = shutil.which("iperf3") is not None
    iperf_part = ""
    if use_iperf:
        iperf_part = r'''
ip netns exec h2-$S iperf3 -s -1 > "$TMPDIR/iperf-server.log" 2>&1 &
srv=$!
sleep 1
if ip netns exec h1-$S iperf3 -c 10.2.0.1 -t "$SECONDS" -P 4 -J > "$TMPDIR/iperf-client.json" 2>"$TMPDIR/iperf-client.err"; then
  python3 - "$TMPDIR/iperf-client.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
bits=d.get("end",{}).get("sum_received",{}).get("bits_per_second")
retrans=d.get("end",{}).get("sum_sent",{}).get("retransmits")
print("iperf\t%.3f\t%s" % ((bits or 0.0), retrans if retrans is not None else ""))
PY
else
  echo "iperf_error\t0\t"
fi
wait "$srv" 2>/dev/null || true
'''
    else:
        iperf_part = r'''
ip netns exec h1-$S ping -i 0.001 -w "$SECONDS" 10.2.0.1 > "$TMPDIR/ping.log" 2>&1 || true
python3 - "$TMPDIR/ping.log" <<'PY'
import re, sys
s=open(sys.argv[1], errors="ignore").read()
tx=rx=loss=None
m=re.search(r"(\d+) packets transmitted, (\d+) received, .*?(\d+(?:\.\d+)?)% packet loss", s)
if m:
    tx=int(m.group(1)); rx=int(m.group(2)); loss=float(m.group(3))
rtt_avg=""
m=re.search(r"rtt min/avg/max/(?:mdev|stddev) = ([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", s)
if m:
    rtt_avg=m.group(2)
print("ping\t%s\t%s\t%s\t%s" % (tx if tx is not None else "", rx if rx is not None else "", loss if loss is not None else "", rtt_avg))
PY
'''
    script = rf'''
set -euo pipefail
S="{suffix}"
SECONDS="{seconds}"
TMPDIR="/tmp/forward-sideeffect-$S"
mkdir -p "$TMPDIR"
cleanup() {{
  ip netns del h1-$S 2>/dev/null || true
  ip netns del r-$S 2>/dev/null || true
  ip netns del h2-$S 2>/dev/null || true
  rm -rf "$TMPDIR"
}}
trap cleanup EXIT

ip netns add h1-$S
ip netns add r-$S
ip netns add h2-$S
ip link add h1v-$S type veth peer name rv1-$S
ip link add h2v-$S type veth peer name rv2-$S
ip link set h1v-$S netns h1-$S
ip link set rv1-$S netns r-$S
ip link set h2v-$S netns h2-$S
ip link set rv2-$S netns r-$S
ip -n h1-$S addr add 10.1.0.1/24 dev h1v-$S
ip -n r-$S  addr add 10.1.0.254/24 dev rv1-$S
ip -n h2-$S addr add 10.2.0.1/24 dev h2v-$S
ip -n r-$S  addr add 10.2.0.254/24 dev rv2-$S
for ns in h1-$S r-$S h2-$S; do ip -n "$ns" link set lo up; done
ip -n h1-$S link set h1v-$S up
ip -n h2-$S link set h2v-$S up
ip -n r-$S link set rv1-$S up
ip -n r-$S link set rv2-$S up
ip netns exec r-$S sysctl -qw net.ipv4.ip_forward=1
ip -n h1-$S route add default via 10.1.0.254
ip -n h2-$S route add default via 10.2.0.254
{iperf_part}
'''
    proc = sh(script, timeout=seconds + 60)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    parts = line.split("\t")
    row = {"rep": rep, "tool": parts[0] if parts else ("iperf" if use_iperf else "ping"), "seconds": seconds}
    if parts and parts[0] == "iperf":
        row["bits_per_second"] = float(parts[1])
        row["retransmits"] = parts[2]
        row["success"] = True
    elif parts and parts[0] == "ping":
        row["tx"] = parts[1]
        row["rx"] = parts[2]
        row["loss_pct"] = parts[3]
        row["rtt_avg_ms"] = parts[4] if len(parts) > 4 else ""
        row["success"] = parts[3] not in {"", "100"}
    else:
        row["success"] = False
        row["raw"] = line
        row["stderr"] = proc.stderr[-500:]
    return row


def lifecycle_bench(rep, count):
    prefix = f"lifeprobe-{os.getpid()}-{rep}"
    script = rf'''
set -euo pipefail
COUNT="{count}"
PREFIX="{prefix}"
cleanup() {{
  ip netns list | awk '{{print $1}}' | grep "^${{PREFIX}}-" | xargs -r -n1 ip netns del 2>/dev/null || true
}}
trap cleanup EXIT
start=$(date +%s.%N)
for i in $(seq 1 "$COUNT"); do
  ns="${{PREFIX}}-${{i}}"
  ip netns add "$ns"
  ip -n "$ns" link add dummy0 type dummy
  ip -n "$ns" link set dummy0 up
  c=$(( (i % 250) + 1 ))
  ip -n "$ns" route replace "192.0.2.${{c}}/32" dev dummy0 metric 100
  ip netns del "$ns"
done
end=$(date +%s.%N)
python3 - <<PY
print("%.6f" % ($end - $start))
PY
'''
    proc = sh(script, timeout=max(300, count * 3))
    return {"rep": rep, "count": count, "seconds": float(proc.stdout.strip().splitlines()[-1])}


def monitor_semantics(rep):
    ns = f"monprobe-{os.getpid()}-{rep}"
    route_file = f"/tmp/{ns}.routes"
    script = rf'''
set -euo pipefail
NS="{ns}"
ROUTE_FILE="{route_file}"
cleanup() {{
  ip netns del "$NS" 2>/dev/null || true
  rm -f "$ROUTE_FILE"
}}
trap cleanup EXIT
ip netns add "$NS"
ip -n "$NS" link set lo up
ip netns exec "$NS" sh -c "timeout 5 ip monitor route > '$ROUTE_FILE'" &
monpid=$!
sleep 1
ip -n "$NS" route replace blackhole 203.0.113.1/32
ip -n "$NS" route del blackhole 203.0.113.1/32
wait "$monpid" 2>/dev/null || true
lines=$(wc -l < "$ROUTE_FILE" 2>/dev/null || echo 0)
adds=$(grep -c '203.0.113.1' "$ROUTE_FILE" 2>/dev/null || true)
echo -e "$lines\t$adds"
cat "$ROUTE_FILE" 2>/dev/null || true
'''
    proc = sh(script, timeout=20)
    first = proc.stdout.splitlines()[0].split("\t")
    lines = int(first[0]) if first and first[0] else 0
    route_mentions = int(first[1]) if len(first) > 1 and first[1] else 0
    return {
        "rep": rep,
        "lines": lines,
        "route_mentions": route_mentions,
        "success": route_mentions >= 2,
        "sample": "\n".join(proc.stdout.splitlines()[1:6]),
    }


def kernel_risk_log(since, out_path):
    cmd = f"journalctl -k --since {since!r} --no-pager | egrep -i {RISK_PATTERN!r} || true"
    proc = sh(cmd, check=False)
    out_path.write_text(proc.stdout)
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--route-n", type=int, default=50000)
    parser.add_argument("--lifecycle-count", type=int, default=500)
    parser.add_argument("--forward-seconds", type=int, default=30)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: run as root", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    kernel = kernel_release()

    initial_risk = kernel_risk_log("boot", out_dir / "kernel_risk_since_boot_before.log")

    route_rows = []
    forward_rows = []
    lifecycle_rows = []
    monitor_rows = []
    errors = []

    for rep in range(1, args.reps + 1):
        rep_started = datetime.now(timezone.utc).isoformat()
        try:
            route_rows.extend(standard_route_bench(rep, args.route_n, out_dir))
        except Exception as exc:
            errors.append({"rep": rep, "bench": "standard_route", "error": str(exc)})
        try:
            forward_rows.append(forwarding_bench(rep, args.forward_seconds))
        except Exception as exc:
            errors.append({"rep": rep, "bench": "forwarding", "error": str(exc)})
        try:
            lifecycle_rows.append(lifecycle_bench(rep, args.lifecycle_count))
        except Exception as exc:
            errors.append({"rep": rep, "bench": "lifecycle", "error": str(exc)})
        try:
            monitor_rows.append(monitor_semantics(rep))
        except Exception as exc:
            errors.append({"rep": rep, "bench": "monitor", "error": str(exc)})
        (out_dir / f"rep_{rep}.done").write_text(rep_started + "\n")

    risk_hits = kernel_risk_log(started, out_dir / "kernel_risk_since_bench.log")

    write_tsv(out_dir / "standard_route.tsv", route_rows, ["rep", "phase", "n", "seconds", "routes"])
    write_tsv(out_dir / "forwarding.tsv", forward_rows, ["rep", "tool", "seconds", "bits_per_second", "retransmits", "tx", "rx", "loss_pct", "rtt_avg_ms", "success"])
    write_tsv(out_dir / "lifecycle.tsv", lifecycle_rows, ["rep", "count", "seconds"])
    write_tsv(out_dir / "monitor.tsv", monitor_rows, ["rep", "lines", "route_mentions", "success", "sample"])

    add_times = [r["seconds"] for r in route_rows if r["phase"] == "add"]
    flush_times = [r["seconds"] for r in route_rows if r["phase"] == "flush"]
    lifecycle_times = [r["seconds"] for r in lifecycle_rows]
    iperf_bps = [r.get("bits_per_second", 0.0) for r in forward_rows if r.get("tool") == "iperf" and r.get("success")]
    ping_loss = [float(r["loss_pct"]) for r in forward_rows if r.get("tool") == "ping" and str(r.get("loss_pct", "")) not in {"", "None"}]
    ping_rtt = [float(r["rtt_avg_ms"]) for r in forward_rows if r.get("tool") == "ping" and str(r.get("rtt_avg_ms", "")) not in {"", "None"}]

    summary = {
        "label": args.label,
        "kernel": kernel,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "settings": vars(args),
        "initial_kernel_risk_hits_since_boot": initial_risk,
        "kernel_risk_hits_since_bench": risk_hits,
        "errors": errors,
        "standard_route_add_seconds": summary_stats(add_times),
        "standard_route_flush_seconds": summary_stats(flush_times),
        "lifecycle_seconds": summary_stats(lifecycle_times),
        "forwarding_iperf_bps": summary_stats(iperf_bps),
        "forwarding_ping_loss_pct": summary_stats(ping_loss),
        "forwarding_ping_rtt_avg_ms": summary_stats(ping_rtt),
        "monitor_successes": sum(1 for r in monitor_rows if r.get("success")),
        "monitor_reps": len(monitor_rows),
        "success": not errors and risk_hits == 0 and all(r.get("success") for r in monitor_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
