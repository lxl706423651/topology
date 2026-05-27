import argparse
import csv
import json
import math
import re
import struct
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


TIME_FMT = "%Y-%m-%d %H:%M:%S"
TIME_FMT_MS = "%Y-%m-%d %H:%M:%S.%f"
MRT_BGP4MP = 16
MRT_BGP4MP_MESSAGE_SUBTYPES_2B_AS = {1, 6}
MRT_BGP4MP_MESSAGE_SUBTYPES_4B_AS = {4, 7}
IPERF_RESULT_NAME_RE = re.compile(
    r"^(?P<client>.+?)_to_(?P<server>.+?)_hop_(?P<hop>\d+)\.txt$"
)
IPERF_LINE_RE = re.compile(
    r"^\[\s*\d+\]\s+"
    r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
    r"(?P<transfer_val>[\d.]+)\s+(?P<transfer_unit>[KMG]?Bytes)\s+"
    r"(?P<bitrate_val>[\d.]+)\s+(?P<bitrate_unit>[KMG]?bits/sec)"
    r"(?:\s+(?P<retr>\d+))?"
    r"(?:\s+(?P<cwnd_val>[\d.]+)\s+(?P<cwnd_unit>[KMG]?Bytes))?"
    r"(?:\s+(?P<role>sender|receiver))?\s*$"
)
UDP_SUMMARY_RE = re.compile(
    r"^\[\s*\d+\]\s+"
    r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
    r"(?P<transfer_val>[\d.]+)\s+(?P<transfer_unit>[KMG]?Bytes)\s+"
    r"(?P<bitrate_val>[\d.]+)\s+(?P<bitrate_unit>[KMG]?bits/sec)\s+"
    r"(?P<jitter_val>[\d.]+)\s+ms\s+"
    r"(?P<lost>\d+)/(?P<total>\d+)\s+\((?P<loss_pct>[\d.]+)%\)\s+"
    r"(?P<role>sender|receiver)\s*$"
)
PING_PACKET_RE = re.compile(
    r"(?P<tx>\d+)\s+packets transmitted,\s+"
    r"(?P<rx>\d+)\s+received,\s+"
    r"(?P<loss_pct>[\d.]+)%\s+packet loss"
)
PING_RTT_RE = re.compile(
    r"rtt min/avg/max/mdev = "
    r"(?P<min>[\d.]+)/(?P<avg>[\d.]+)/(?P<max>[\d.]+)/(?P<mdev>[\d.]+)\s+ms"
)
RTNL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<comm>\S+)\s+"
    r"(?P<op>\S+)\s+"
    r"(?P<container>.+?)\s+\((?P<inode>\d+)\)\s+"
    r"(?P<wait>\d+)\s+(?P<hold>\d+)\s*$"
)
TRACE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<comm>\S+)\s+"
    r"(?P<container>.+?)\s+\((?P<inode>\d+)\)\s+"
    r"(?P<op>[A-Z/]+)\s*$"
)


def parse_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_mean(values):
    return mean(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(values[lower])
    lower_val = values[lower]
    upper_val = values[upper]
    return float(lower_val + (upper_val - lower_val) * (pos - lower))


def format_timestamp_from_epoch(epoch_seconds):
    return datetime.fromtimestamp(epoch_seconds).strftime(TIME_FMT)


def bytes_to_mb(value):
    return value / (1024 * 1024)


def bits_to_gbps(value):
    return value / 1_000_000_000


def convert_bytes(value, unit):
    factors = {
        "Bytes": 1,
        "KBytes": 1024,
        "MBytes": 1024 ** 2,
        "GBytes": 1024 ** 3,
    }
    return float(value) * factors[unit]


def convert_bits_per_second(value, unit):
    factors = {
        "bits/sec": 1,
        "Kbits/sec": 1_000,
        "Mbits/sec": 1_000_000,
        "Gbits/sec": 1_000_000_000,
    }
    return float(value) * factors[unit]


def parse_host_mem_csvs(memory_dir: Path):
    results = []
    for path in sorted(memory_dir.glob("host_mem_*.csv")):
        rows = parse_csv(path)
        if not rows:
            continue
        row = rows[0]
        results.append(
            {
                "file": path.name,
                "timestamp": row["Timestamp"],
                "nodes": int(row["Nodes"]),
                "used_mem_mb": float(row["Used_Mem_MB"]),
                "slab_mem_kb": float(row["Slab_Mem_KB"]),
            }
        )
    return results


def summarize_container_memory(path: Path):
    rows = parse_csv(path)
    if not rows:
        return None
    total_mem_values = [float(row["Total_Mem_MB"]) for row in rows]
    bird_mem_values = [float(row["BIRD_Routing_Mem_MB"]) for row in rows]
    other_mem_values = [float(row["Other_Mem_MB"]) for row in rows]
    top10 = sorted(
        rows,
        key=lambda row: float(row["BIRD_Routing_Mem_MB"]),
        reverse=True,
    )[:10]
    return {
        "file": path.name,
        "containers": len(rows),
        "total_mem_mb_sum": round(sum(total_mem_values), 2),
        "bird_mem_mb_sum": round(sum(bird_mem_values), 2),
        "other_mem_mb_sum": round(sum(other_mem_values), 2),
        "total_mem_mb_avg": round(safe_mean(total_mem_values), 4),
        "bird_mem_mb_avg": round(safe_mean(bird_mem_values), 4),
        "other_mem_mb_avg": round(safe_mean(other_mem_values), 4),
        "top10_bird_mem": [
            {
                "container": row["Container_Name"],
                "bird_routing_mem_mb": float(row["BIRD_Routing_Mem_MB"]),
                "total_mem_mb": float(row["Total_Mem_MB"]),
            }
            for row in top10
        ],
    }


def parse_memory_snapshots(host_memory_dir: Path):
    results = []
    for path in sorted(host_memory_dir.glob("memory_snapshot_*.csv")):
        summary = summarize_container_memory(path)
        if summary is not None:
            results.append(summary)
    return results


def aggregate_memory_snapshots(memory_snapshots):
    if not memory_snapshots:
        return {}
    total_avgs = [item["total_mem_mb_avg"] for item in memory_snapshots]
    bird_avgs = [item["bird_mem_mb_avg"] for item in memory_snapshots]
    total_sums = [item["total_mem_mb_sum"] for item in memory_snapshots]
    bird_sums = [item["bird_mem_mb_sum"] for item in memory_snapshots]
    return {
        "snapshot_count": len(memory_snapshots),
        "avg_total_mem_mb_across_snapshots": round(safe_mean(total_avgs), 4),
        "avg_bird_mem_mb_across_snapshots": round(safe_mean(bird_avgs), 4),
        "avg_total_mem_sum_mb_across_snapshots": round(safe_mean(total_sums), 2),
        "avg_bird_mem_sum_mb_across_snapshots": round(safe_mean(bird_sums), 2),
    }


def parse_docker_build(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    built_count = len(re.findall(r"\bBuilt\b", text))
    runtime_match = re.search(r"Total runtime:\s*([^\r\n]+)", text)
    return {
        "file": log_path.name,
        "built_entries": built_count,
        "runtime": runtime_match.group(1).strip() if runtime_match else None,
    }


def parse_docker_up(log_path: Path):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    started = sum(1 for line in lines if " Container " in line and " Started" in line)
    networks_created = sum(1 for line in lines if " Network " in line and " Created" in line)
    runtime = None
    script_start = None
    script_end = None
    for line in lines:
        if script_start is None:
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ", line)
            if match:
                script_start = match.group(1)
        runtime_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Total runtime:\s*(.+)", line)
        if runtime_match:
            script_end = runtime_match.group(1)
            runtime = runtime_match.group(2).strip()
    return {
        "file": log_path.name,
        "script_start": script_start,
        "script_end": script_end,
        "runtime": runtime,
        "started_containers": started,
        "created_networks": networks_created,
    }


def parse_phase_summaries(session_log: Path):
    lines = session_log.read_text(encoding="utf-8", errors="replace").splitlines()
    phases = []
    for idx, line in enumerate(lines):
        if "Failure" not in line and "Recovery" not in line:
            continue
        phase_name = "failure" if "Failure" in line else "recovery"
        segment = "\n".join(lines[idx: idx + 8])
        t_start_match = re.search(r"T_start\):\s*([0-9:.]+)", segment)
        t_end_match = re.search(r"T_end\)\s*:?\s*([0-9:.]+)", segment)
        fib_match = re.search(r"FIB .*?:\s*(\d+)", segment)
        duration_match = re.search(r"([0-9.]+)\s*秒", segment)
        if not all([t_start_match, t_end_match, fib_match, duration_match]):
            continue
        phases.append(
            {
                "session_file": session_log.name,
                "phase_name": phase_name,
                "t_start": t_start_match.group(1),
                "t_end": t_end_match.group(1),
                "fib_updates": int(fib_match.group(1)),
                "duration_seconds": float(duration_match.group(1)),
            }
        )
    return phases


def parse_all_phase_summaries(convergence_dir: Path):
    all_phases = []
    for path in sorted(convergence_dir.glob("lifecycle_session_*.log")):
        all_phases.extend(parse_phase_summaries(path))
    return all_phases


def summarize_phases(phases):
    grouped = defaultdict(list)
    for item in phases:
        grouped[item["phase_name"]].append(item)
    summary = {}
    for phase_name, items in grouped.items():
        durations = [item["duration_seconds"] for item in items]
        fib_updates = [item["fib_updates"] for item in items]
        slowest = max(items, key=lambda item: item["duration_seconds"])
        summary[phase_name] = {
            "count": len(items),
            "avg_duration_seconds": round(safe_mean(durations), 3),
            "max_duration_seconds": round(max(durations), 3),
            "avg_fib_updates": round(safe_mean(fib_updates), 2),
            "max_fib_updates": max(fib_updates),
            "slowest_session_file": slowest["session_file"],
        }
    return summary


def parse_trace_log(trace_log: Path):
    counts_by_op = Counter()
    counts_by_container = Counter()
    first_ts = None
    last_ts = None
    total_events = 0
    for line in trace_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACE_RE.match(line.strip())
        if not match:
            continue
        total_events += 1
        ts = datetime.strptime(match.group("ts"), TIME_FMT_MS)
        first_ts = ts if first_ts is None else min(first_ts, ts)
        last_ts = ts if last_ts is None else max(last_ts, ts)
        counts_by_op[match.group("op")] += 1
        counts_by_container[match.group("container")] += 1
    return {
        "file": trace_log.name,
        "total_events": total_events,
        "first_event": first_ts.strftime(TIME_FMT_MS)[:-3] if first_ts else None,
        "last_event": last_ts.strftime(TIME_FMT_MS)[:-3] if last_ts else None,
        "event_span_seconds": round((last_ts - first_ts).total_seconds(), 3) if first_ts and last_ts else None,
        "counts_by_operation": dict(counts_by_op),
        "top10_containers_by_events": counts_by_container.most_common(10),
    }


def aggregate_trace_logs(trace_entries):
    counts_by_op = Counter()
    total_events = 0
    for entry in trace_entries:
        total_events += entry["total_events"]
        counts_by_op.update(entry["counts_by_operation"])
    return {
        "file_count": len(trace_entries),
        "total_events": total_events,
        "counts_by_operation": dict(counts_by_op),
    }


def parse_rtnl_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RTNL_RE.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "ts": match.group("ts"),
                "ts_dt": datetime.strptime(match.group("ts"), TIME_FMT_MS),
                "pid": match.group("pid"),
                "comm": match.group("comm"),
                "op": match.group("op"),
                "container": match.group("container"),
                "inode": match.group("inode"),
                "wait_us": int(match.group("wait")),
                "hold_us": int(match.group("hold")),
            }
        )
    return rows


def pair_rtnl_rows(rows):
    paired = []
    idx = 0
    while idx < len(rows):
        current = rows[idx]
        pair_rows = [current]
        if idx + 1 < len(rows):
            nxt = rows[idx + 1]
            same_pid = current["pid"] == nxt["pid"]
            same_container = current["container"] == nxt["container"]
            same_inode = current["inode"] == nxt["inode"]
            close_in_time = abs((nxt["ts_dt"] - current["ts_dt"]).total_seconds()) <= 0.01
            if same_pid and same_container and same_inode and close_in_time:
                pair_rows.append(nxt)
                idx += 1
        paired.append(
            {
                "ts": pair_rows[0]["ts"],
                "pid": pair_rows[0]["pid"],
                "container": pair_rows[0]["container"],
                "ops": [row["op"] for row in pair_rows],
                "pair_size": len(pair_rows),
                "wait_sum_us": sum(row["wait_us"] for row in pair_rows),
                "hold_sum_us": sum(row["hold_us"] for row in pair_rows),
                "total_lock_us": sum(row["wait_us"] + row["hold_us"] for row in pair_rows),
            }
        )
        idx += 1
    return paired


def build_log2_distribution(values):
    if not values:
        return []
    max_value = max(values)
    buckets = []
    start = 0
    end = 1
    while start <= max_value:
        count = sum(1 for value in values if start <= value <= end)
        buckets.append({"start_us": start, "end_us": end, "count": count})
        if end == 1:
            start = 2
            end = 3
        else:
            start = end + 1
            end = (end + 1) * 2 - 1
    return buckets


def compress_distribution(distribution):
    return [
        f"{bucket['start_us']}-{bucket['end_us']}us:{bucket['count']}"
        for bucket in distribution
        if bucket["count"] > 0
    ]


def summarize_numeric(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "avg": round(safe_mean(values), 2),
        "median": round(median(values), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": round(max(values), 2),
        "sum": round(sum(values), 2),
    }


def parse_rtnl_log(path: Path):
    raw_rows = parse_rtnl_rows(path)
    if not raw_rows:
        return {"file": path.name, "raw_rows": 0, "paired_rows": 0}
    paired_rows = pair_rtnl_rows(raw_rows)
    pair_waits = [row["wait_sum_us"] for row in paired_rows]
    pair_holds = [row["hold_sum_us"] for row in paired_rows]
    pair_totals = [row["total_lock_us"] for row in paired_rows]
    by_container_wait = defaultdict(int)
    for row in paired_rows:
        by_container_wait[row["container"]] += row["wait_sum_us"]
    max_wait_row = max(paired_rows, key=lambda row: row["wait_sum_us"])
    max_total_row = max(paired_rows, key=lambda row: row["total_lock_us"])
    return {
        "file": path.name,
        "raw_rows": len(raw_rows),
        "paired_rows": len(paired_rows),
        "single_rows_after_pairing": sum(1 for row in paired_rows if row["pair_size"] == 1),
        "wait_us_stats": summarize_numeric(pair_waits),
        "hold_us_stats": summarize_numeric(pair_holds),
        "total_lock_us_stats": summarize_numeric(pair_totals),
        "max_wait_entry": max_wait_row,
        "max_total_lock_entry": max_total_row,
        "top10_containers_by_wait_sum_us": sorted(
            by_container_wait.items(), key=lambda item: item[1], reverse=True
        )[:10],
        "wait_distribution": build_log2_distribution(pair_waits),
        "hold_distribution": build_log2_distribution(pair_holds),
        "total_lock_distribution": build_log2_distribution(pair_totals),
    }


def parse_csw_log(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                if len(parts) >= 2 and parts[1] in {"AM", "PM"}:
                    if len(parts) < 6:
                        continue
                    row = {
                        "timestamp": f"{parts[0]} {parts[1]}",
                        "pid": parts[3],
                        "cswch_s": float(parts[4]) if parts[4] else 0.0,
                        "nvcswch_s": float(parts[5]) if parts[5] else 0.0,
                    }
                else:
                    row = {
                        "timestamp": parts[0],
                        "pid": parts[2] if len(parts) > 2 else "",
                        "cswch_s": float(parts[3]) if len(parts) > 3 and parts[3] else 0.0,
                        "nvcswch_s": float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
                    }
            except ValueError:
                continue
            rows.append(row)
    if not rows:
        return {"file": path.name, "rows": 0}
    cswch_values = [row["cswch_s"] for row in rows]
    nvcswch_values = [row["nvcswch_s"] for row in rows]
    max_row = max(rows, key=lambda row: row["cswch_s"])
    return {
        "file": path.name,
        "rows": len(rows),
        "avg_cswch_s": round(safe_mean(cswch_values), 3),
        "avg_nvcswch_s": round(safe_mean(nvcswch_values), 3),
        "p95_cswch_s": round(percentile(cswch_values, 0.95), 3),
        "max_cswch_s": max_row["cswch_s"],
        "max_cswch_pid": max_row["pid"],
        "max_cswch_timestamp": max_row["timestamp"],
    }


def count_prefixes(data):
    idx = 0
    count = 0
    length = len(data)
    while idx < length:
        prefix_len = data[idx]
        idx += 1
        octets = (prefix_len + 7) // 8
        idx += octets
        if idx <= length:
            count += 1
        else:
            break
    return count


def get_mrt_bgp_message_offset(subtype, payload):
    if subtype in MRT_BGP4MP_MESSAGE_SUBTYPES_2B_AS:
        as_len = 2
    elif subtype in MRT_BGP4MP_MESSAGE_SUBTYPES_4B_AS:
        as_len = 4
    else:
        return None
    if len(payload) < (2 * as_len + 4):
        return None
    afi_offset = 2 * as_len + 2
    afi = struct.unpack(">H", payload[afi_offset:afi_offset + 2])[0]
    if afi == 1:
        ip_len = 4
    elif afi == 2:
        ip_len = 16
    else:
        return None
    offset = 2 * as_len + 4 + 2 * ip_len
    if offset + 19 > len(payload):
        return None
    if payload[offset:offset + 16] != b"\xff" * 16:
        return None
    return offset


def parse_bgp_update_message(bgp_message):
    if len(bgp_message) < 23:
        return None
    message_len = struct.unpack(">H", bgp_message[16:18])[0]
    if message_len > len(bgp_message) or message_len < 19:
        return None
    message_type = bgp_message[18]
    if message_type != 2:
        return {"message_type": message_type}
    body = bgp_message[19:message_len]
    if len(body) < 4:
        return {"message_type": message_type}
    withdrawn_len = struct.unpack(">H", body[:2])[0]
    withdrawn_end = 2 + withdrawn_len
    if withdrawn_end + 2 > len(body):
        return {"message_type": message_type}
    withdrawn = body[2:withdrawn_end]
    attrs_len = struct.unpack(">H", body[withdrawn_end:withdrawn_end + 2])[0]
    attrs_end = withdrawn_end + 2 + attrs_len
    if attrs_end > len(body):
        return {"message_type": message_type}
    nlri = body[attrs_end:]
    return {
        "message_type": message_type,
        "withdrawn_prefixes": count_prefixes(withdrawn),
        "announced_prefixes": count_prefixes(nlri),
    }


def parse_mrt_file(path: Path):
    record_count = 0
    bgp_message_count = 0
    update_message_count = 0
    updates_per_second = Counter()
    prefix_changes_per_second = Counter()
    announced_prefixes_total = 0
    withdrawn_prefixes_total = 0
    with path.open("rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) < 12:
                break
            timestamp, record_type, subtype, length = struct.unpack(">IHHI", header)
            payload = f.read(length)
            if len(payload) < length:
                break
            record_count += 1
            if record_type != MRT_BGP4MP:
                continue
            offset = get_mrt_bgp_message_offset(subtype, payload)
            if offset is None:
                continue
            bgp_message = payload[offset:]
            parsed = parse_bgp_update_message(bgp_message)
            if parsed is None:
                continue
            bgp_message_count += 1
            if parsed.get("message_type") != 2:
                continue
            update_message_count += 1
            updates_per_second[timestamp] += 1
            announced = parsed.get("announced_prefixes", 0)
            withdrawn = parsed.get("withdrawn_prefixes", 0)
            announced_prefixes_total += announced
            withdrawn_prefixes_total += withdrawn
            prefix_changes_per_second[timestamp] += announced + withdrawn
    peak_update_second, peak_update_count = (None, 0)
    if updates_per_second:
        peak_update_second, peak_update_count = max(
            updates_per_second.items(), key=lambda item: item[1]
        )
    peak_prefix_second, peak_prefix_count = (None, 0)
    if prefix_changes_per_second:
        peak_prefix_second, peak_prefix_count = max(
            prefix_changes_per_second.items(), key=lambda item: item[1]
        )
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "size_kb": round(path.stat().st_size / 1024, 2),
        "record_count": record_count,
        "bgp_message_count": bgp_message_count,
        "update_message_count": update_message_count,
        "announced_prefixes_total": announced_prefixes_total,
        "withdrawn_prefixes_total": withdrawn_prefixes_total,
        "peak_update_second": format_timestamp_from_epoch(peak_update_second) if peak_update_second else None,
        "peak_update_count": peak_update_count,
        "peak_prefix_change_second": format_timestamp_from_epoch(peak_prefix_second) if peak_prefix_second else None,
        "peak_prefix_change_count": peak_prefix_count,
    }


def parse_mrt_files(mrt_dir: Path):
    return [parse_mrt_file(path) for path in sorted(mrt_dir.glob("*.mrt"))]


def find_iperf_task_file(result_dir: Path):
    candidates = [
        result_dir / "iperf_task.json",
        result_dir / "iperf_tasks.json",
        result_dir.parent / "iperf_task.json",
        result_dir.parent / "iperf_tasks.json",
        Path.cwd() / "iperf_task.json",
        Path.cwd() / "iperf_tasks.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_iperf_task_map(task_file: Path):
    if task_file is None or not task_file.exists():
        return {}
    data = json.loads(task_file.read_text(encoding="utf-8"))
    task_map = {}
    for item in data:
        task_map[(item["client_node"], item["server_node"])] = item
    return task_map


def parse_iperf_result_file(path: Path, task_map):
    match = IPERF_RESULT_NAME_RE.match(path.name)
    if not match:
        return None
    client = match.group("client")
    server = match.group("server")
    fallback_hop = int(match.group("hop"))
    task = task_map.get((client, server), {})
    tcp_intervals = []
    sender_summary = None
    receiver_summary = None
    udp_sender_summary = None
    udp_receiver_summary = None
    udp_error = None
    ping_packet_stats = None
    ping_rtt_stats = None
    current_section = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if "=== 1. TCP Bandwidth" in line:
            current_section = "tcp"
            continue
        if "=== 2. UDP Jitter" in line:
            current_section = "udp"
            continue
        if "=== 3. ICMP Latency" in line:
            current_section = "ping"
            continue
        line_match = IPERF_LINE_RE.match(line)
        if not line_match:
            if current_section == "udp":
                udp_match = UDP_SUMMARY_RE.match(line)
                if udp_match:
                    udp_parsed = {
                        "start": float(udp_match.group("start")),
                        "end": float(udp_match.group("end")),
                        "transfer_bytes": convert_bytes(udp_match.group("transfer_val"), udp_match.group("transfer_unit")),
                        "bitrate_bps": convert_bits_per_second(udp_match.group("bitrate_val"), udp_match.group("bitrate_unit")),
                        "jitter_ms": float(udp_match.group("jitter_val")),
                        "lost_datagrams": int(udp_match.group("lost")),
                        "total_datagrams": int(udp_match.group("total")),
                        "loss_pct": float(udp_match.group("loss_pct")),
                        "role": udp_match.group("role"),
                    }
                    if udp_parsed["role"] == "sender":
                        udp_sender_summary = udp_parsed
                    else:
                        udp_receiver_summary = udp_parsed
                    continue
                if "iperf3: error" in line:
                    udp_error = line
                    continue
            if current_section == "ping":
                packet_match = PING_PACKET_RE.search(line)
                if packet_match:
                    ping_packet_stats = {
                        "tx": int(packet_match.group("tx")),
                        "rx": int(packet_match.group("rx")),
                        "loss_pct": float(packet_match.group("loss_pct")),
                    }
                    continue
                rtt_match = PING_RTT_RE.search(line)
                if rtt_match:
                    ping_rtt_stats = {
                        "min_ms": float(rtt_match.group("min")),
                        "avg_ms": float(rtt_match.group("avg")),
                        "max_ms": float(rtt_match.group("max")),
                        "mdev_ms": float(rtt_match.group("mdev")),
                    }
                    continue
            continue
        parsed = {
            "start": float(line_match.group("start")),
            "end": float(line_match.group("end")),
            "transfer_bytes": convert_bytes(line_match.group("transfer_val"), line_match.group("transfer_unit")),
            "bitrate_bps": convert_bits_per_second(line_match.group("bitrate_val"), line_match.group("bitrate_unit")),
            "retr": int(line_match.group("retr")) if line_match.group("retr") else 0,
            "cwnd_bytes": convert_bytes(line_match.group("cwnd_val"), line_match.group("cwnd_unit"))
            if line_match.group("cwnd_val") and line_match.group("cwnd_unit")
            else None,
            "role": line_match.group("role"),
        }
        if current_section == "tcp":
            if parsed["role"] == "sender":
                sender_summary = parsed
            elif parsed["role"] == "receiver":
                receiver_summary = parsed
            else:
                tcp_intervals.append(parsed)
    if sender_summary is None:
        return None
    interval_bitrates = [bits_to_gbps(item["bitrate_bps"]) for item in tcp_intervals]
    interval_retr = [item["retr"] for item in tcp_intervals]
    interval_cwnd = [bytes_to_mb(item["cwnd_bytes"]) for item in tcp_intervals if item["cwnd_bytes"] is not None]
    return {
        "file": path.name,
        "client_node": client,
        "server_node": server,
        "type": task.get("type") or "Unknown",
        "hop_count": int(task.get("hop_count", fallback_hop)),
        "as_status": task.get("as_status") or "Unknown",
        "sender_transfer_gb": round(sender_summary["transfer_bytes"] / (1024 ** 3), 3),
        "sender_bitrate_gbps": round(bits_to_gbps(sender_summary["bitrate_bps"]), 3),
        "receiver_transfer_gb": round(receiver_summary["transfer_bytes"] / (1024 ** 3), 3) if receiver_summary else None,
        "receiver_bitrate_gbps": round(bits_to_gbps(receiver_summary["bitrate_bps"]), 3) if receiver_summary else None,
        "retransmits": sender_summary["retr"],
        "interval_count": len(tcp_intervals),
        "interval_bitrate_avg_gbps": round(safe_mean(interval_bitrates), 3) if interval_bitrates else None,
        "interval_bitrate_min_gbps": round(min(interval_bitrates), 3) if interval_bitrates else None,
        "interval_bitrate_max_gbps": round(max(interval_bitrates), 3) if interval_bitrates else None,
        "interval_bitrate_std_gbps": round(pstdev(interval_bitrates), 3) if len(interval_bitrates) > 1 else 0.0,
        "interval_retr_total": sum(interval_retr),
        "interval_retr_peak": max(interval_retr) if interval_retr else 0,
        "avg_cwnd_mb": round(safe_mean(interval_cwnd), 3) if interval_cwnd else None,
        "udp_success": udp_receiver_summary is not None or udp_sender_summary is not None,
        "udp_error": udp_error,
        "udp_sender_bitrate_mbps": round(udp_sender_summary["bitrate_bps"] / 1_000_000, 3) if udp_sender_summary else None,
        "udp_receiver_bitrate_mbps": round(udp_receiver_summary["bitrate_bps"] / 1_000_000, 3) if udp_receiver_summary else None,
        "udp_jitter_ms": round(udp_receiver_summary["jitter_ms"], 6) if udp_receiver_summary else None,
        "udp_loss_pct": round(udp_receiver_summary["loss_pct"], 6) if udp_receiver_summary else None,
        "udp_lost_datagrams": udp_receiver_summary["lost_datagrams"] if udp_receiver_summary else None,
        "udp_total_datagrams": udp_receiver_summary["total_datagrams"] if udp_receiver_summary else None,
        "ping_success": ping_packet_stats is not None and ping_rtt_stats is not None,
        "ping_packet_loss_pct": ping_packet_stats["loss_pct"] if ping_packet_stats else None,
        "ping_tx": ping_packet_stats["tx"] if ping_packet_stats else None,
        "ping_rx": ping_packet_stats["rx"] if ping_packet_stats else None,
        "ping_rtt_min_ms": ping_rtt_stats["min_ms"] if ping_rtt_stats else None,
        "ping_rtt_avg_ms": ping_rtt_stats["avg_ms"] if ping_rtt_stats else None,
        "ping_rtt_max_ms": ping_rtt_stats["max_ms"] if ping_rtt_stats else None,
        "ping_rtt_mdev_ms": ping_rtt_stats["mdev_ms"] if ping_rtt_stats else None,
    }


def parse_iperf_results(result_dir: Path):
    task_file = find_iperf_task_file(result_dir)
    task_map = load_iperf_task_map(task_file)
    result_dirs = [result_dir / "iperfResult", result_dir / "iperf"]
    files = []
    for directory in result_dirs:
        if directory.exists():
            files.extend(sorted(directory.glob("*.txt")))
    entries = []
    for path in files:
        parsed = parse_iperf_result_file(path, task_map)
        if parsed is not None:
            entries.append(parsed)
    return {
        "task_file": str(task_file) if task_file else None,
        "files_scanned": len(files),
        "entries": entries,
    }


def summarize_iperf(entries):
    if not entries:
        return {
            "run_count": 0,
            "by_as_status": [],
            "by_as_status_and_hop": [],
            "best_sender_run": None,
            "worst_sender_run": None,
        }

    def build_group(rows):
        sender_rates = [row["sender_bitrate_gbps"] for row in rows]
        receiver_rates = [row["receiver_bitrate_gbps"] for row in rows if row["receiver_bitrate_gbps"] is not None]
        retr_values = [row["retransmits"] for row in rows]
        udp_rows = [row for row in rows if row["udp_success"]]
        udp_jitter_values = [row["udp_jitter_ms"] for row in udp_rows if row["udp_jitter_ms"] is not None]
        udp_loss_values = [row["udp_loss_pct"] for row in udp_rows if row["udp_loss_pct"] is not None]
        ping_rows = [row for row in rows if row["ping_success"]]
        ping_avg_values = [row["ping_rtt_avg_ms"] for row in ping_rows if row["ping_rtt_avg_ms"] is not None]
        ping_max_values = [row["ping_rtt_max_ms"] for row in ping_rows if row["ping_rtt_max_ms"] is not None]
        return {
            "count": len(rows),
            "avg_sender_gbps": round(safe_mean(sender_rates), 3),
            "min_sender_gbps": round(min(sender_rates), 3),
            "max_sender_gbps": round(max(sender_rates), 3),
            "avg_receiver_gbps": round(safe_mean(receiver_rates), 3) if receiver_rates else None,
            "avg_retransmits": round(safe_mean(retr_values), 2),
            "total_retransmits": sum(retr_values),
            "udp_success_count": len(udp_rows),
            "udp_failure_count": len(rows) - len(udp_rows),
            "avg_udp_jitter_ms": round(safe_mean(udp_jitter_values), 6) if udp_jitter_values else None,
            "avg_udp_loss_pct": round(safe_mean(udp_loss_values), 6) if udp_loss_values else None,
            "max_udp_loss_pct": round(max(udp_loss_values), 6) if udp_loss_values else None,
            "ping_success_count": len(ping_rows),
            "avg_ping_rtt_ms": round(safe_mean(ping_avg_values), 6) if ping_avg_values else None,
            "max_ping_rtt_ms": round(max(ping_max_values), 6) if ping_max_values else None,
        }

    by_status = defaultdict(list)
    by_status_hop = defaultdict(list)
    for row in entries:
        by_status[row.get("as_status", "Unknown")].append(row)
        by_status_hop[(row.get("as_status", "Unknown"), row.get("hop_count"))].append(row)

    by_as_status = []
    for status, rows in sorted(by_status.items()):
        item = {"as_status": status}
        item.update(build_group(rows))
        by_as_status.append(item)

    by_as_status_and_hop = []
    for (status, hop_count), rows in sorted(by_status_hop.items(), key=lambda item: (item[0][0], item[0][1])):
        item = {"as_status": status, "hop_count": hop_count}
        item.update(build_group(rows))
        by_as_status_and_hop.append(item)

    return {
        "run_count": len(entries),
        "by_as_status": by_as_status,
        "by_as_status_and_hop": by_as_status_and_hop,
        "best_sender_run": max(entries, key=lambda row: row["sender_bitrate_gbps"]),
        "worst_sender_run": min(entries, key=lambda row: row["sender_bitrate_gbps"]),
        "best_ping_run": min(
            [row for row in entries if row["ping_rtt_avg_ms"] is not None],
            key=lambda row: row["ping_rtt_avg_ms"],
            default=None,
        ),
        "worst_ping_run": max(
            [row for row in entries if row["ping_rtt_avg_ms"] is not None],
            key=lambda row: row["ping_rtt_avg_ms"],
            default=None,
        ),
        "worst_udp_loss_run": max(
            [row for row in entries if row["udp_loss_pct"] is not None],
            key=lambda row: row["udp_loss_pct"],
            default=None,
        ),
        "worst_udp_jitter_run": max(
            [row for row in entries if row["udp_jitter_ms"] is not None],
            key=lambda row: row["udp_jitter_ms"],
            default=None,
        ),
    }


def build_timeline(host_mem, docker_up, phases):
    items = []
    if host_mem:
        items.append(("step0_host_memory", host_mem[0]["timestamp"]))
    if docker_up.get("script_start"):
        items.append(("step2_docker_up_start", docker_up["script_start"]))
    if docker_up.get("script_end"):
        items.append(("step2_docker_up_end", docker_up["script_end"]))
    for phase in phases:
        items.append((f"{phase['session_file']}_{phase['phase_name']}_t_start", phase["t_start"]))
        items.append((f"{phase['session_file']}_{phase['phase_name']}_t_end", phase["t_end"]))
    if len(host_mem) > 1:
        items.append(("step5_host_memory", host_mem[1]["timestamp"]))
    if len(host_mem) > 2:
        items.append(("step10_host_memory", host_mem[2]["timestamp"]))
    return items


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def generate_report(data):
    host_mem = data["host_mem"]
    memory_snapshots = data["memory_snapshots"]
    memory_aggregate = data["memory_aggregate"]
    docker_build = data["docker_build"]
    docker_up = data["docker_up"]
    phases = data["phases"]
    phase_summary = data["phase_summary"]
    trace_aggregate = data["trace_aggregate"]
    mrt_files = data["mrt_files"]
    iperf_summary = data["iperf_summary"]
    rtnl_entries = data["rtnl"]
    csw_entries = data["csw"]

    lines = []
    lines.append(f"# 日志分析报告: {data['result_dir'].name}")
    lines.append("")
    lines.append("## 1. 目录识别")
    lines.append("")
    lines.append(f"- docker build 日志: `{docker_build['file']}`")
    lines.append(f"- docker up 日志: `{docker_up['file']}`")
    lines.append(f"- lifecycle_session 日志: `{len(list((data['result_dir'] / 'convergence').glob('lifecycle_session_*.log')))} 个`")
    lines.append(f"- lifecycle_trace 日志: `{len(data['trace_logs'])} 个`")
    lines.append(f"- host memory 快照: `{len(host_mem)} 份`")
    lines.append(f"- 容器内存快照: `{len(memory_snapshots)} 份`")
    lines.append(f"- seed profiler: `rtnl_trace` {len(rtnl_entries)} 个, `csw_trace` {len(csw_entries)} 个")
    lines.append(f"- MRT 文件: `{len(mrt_files)} 个`")
    lines.append(f"- iperf 结果文件: `{iperf_summary['run_count']} 个`")
    lines.append("")

    lines.append("## 2. 流程关键结果")
    lines.append("")
    if host_mem:
        lines.append(f"- 步骤 0 宿主机内存: `{host_mem[0]['used_mem_mb']:.0f} MB`, slab `{host_mem[0]['slab_mem_kb']:.0f} KB`")
    lines.append(f"- 步骤 1 buildTest1.sh: 构建时长 `{docker_build['runtime']}`")
    lines.append(f"- 步骤 2 docker_up.sh: 启动时长 `{docker_up['runtime']}`, 启动容器 `{docker_up['started_containers']}` 个")
    if memory_snapshots:
        first_snapshot = memory_snapshots[0]
        lines.append(
            f"- 步骤 5 容器内存快照平均值: Total_Mem_MB 平均 `{first_snapshot['total_mem_mb_avg']:.4f}`, "
            f"BIRD_Routing_Mem_MB 平均 `{first_snapshot['bird_mem_mb_avg']:.4f}`"
        )
    for item in mrt_files:
        lines.append(
            f"- 步骤 6 MRT `{item['file']}`: BGP UPDATE 报文 `{item['update_message_count']}` 条, 按秒峰值 `{item['peak_update_count']}` 条 @ `{item['peak_update_second']}`"
        )
    if len(host_mem) > 2:
        lines.append(f"- 步骤 10 宿主机内存: `{host_mem[2]['used_mem_mb']:.0f} MB`, slab `{host_mem[2]['slab_mem_kb']:.0f} KB`")
    lines.append(f"- 步骤 11 iperf 已解析 `{iperf_summary['run_count']}` 条任务结果")
    lines.append("")

    lines.append("## 3. 收敛日志")
    lines.append("")
    for phase in phases:
        phase_cn = "节点失效" if phase["phase_name"] == "failure" else "节点恢复"
        lines.append(
            f"- `{phase['session_file']}` {phase_cn}: T_start `{phase['t_start']}`, T_end `{phase['t_end']}`, "
            f"FIB 更新 `{phase['fib_updates']}`, 绝对收敛时间 `{phase['duration_seconds']:.3f}s`"
        )
    if phase_summary:
        lines.append("")
        for phase_name in ("failure", "recovery"):
            if phase_name not in phase_summary:
                continue
            item = phase_summary[phase_name]
            phase_cn = "失效阶段" if phase_name == "failure" else "恢复阶段"
            lines.append(
                f"- {phase_cn}汇总: `{item['count']}` 次, 平均收敛 `{item['avg_duration_seconds']:.3f}s`, "
                f"最大收敛 `{item['max_duration_seconds']:.3f}s`, 平均 FIB 更新 `{item['avg_fib_updates']}`"
            )
    lines.append("")

    lines.append("## 4. MRT 解析")
    lines.append("")
    if mrt_files:
        for item in mrt_files:
            lines.append(
                f"- `{item['file']}`: BGP 消息 `{item['bgp_message_count']}`, UPDATE 消息 `{item['update_message_count']}`, "
                f"宣告前缀 `{item['announced_prefixes_total']}`, 撤销前缀 `{item['withdrawn_prefixes_total']}`"
            )
            lines.append(
                f"  UPDATE 峰值: `{item['peak_update_count']}` 条/秒 @ `{item['peak_update_second']}`; "
                f"前缀变化峰值: `{item['peak_prefix_change_count']}` 条/秒 @ `{item['peak_prefix_change_second']}`"
            )
    else:
        lines.append("- 未发现 MRT 文件。")
    lines.append("")

    lines.append("## 5. iperf 分类统计")
    lines.append("")
    if iperf_summary["run_count"] > 0:
        for item in iperf_summary["by_as_status"]:
            receiver_text = f", 平均 receiver 带宽 `{item['avg_receiver_gbps']:.3f} Gbps`" if item["avg_receiver_gbps"] is not None else ""
            udp_text = (
                f", UDP 成功 `{item['udp_success_count']}` 条, 平均抖动 `{item['avg_udp_jitter_ms']:.6f} ms`, 平均丢包 `{item['avg_udp_loss_pct']:.6f}%`"
                if item["avg_udp_jitter_ms"] is not None or item["avg_udp_loss_pct"] is not None
                else f", UDP 成功 `{item['udp_success_count']}` 条"
            )
            ping_text = (
                f", ping 平均 RTT `{item['avg_ping_rtt_ms']:.6f} ms`"
                if item["avg_ping_rtt_ms"] is not None
                else ""
            )
            lines.append(
                f"- `{item['as_status']}`: `{item['count']}` 条, 平均 sender 带宽 `{item['avg_sender_gbps']:.3f} Gbps`{receiver_text}, 平均重传 `{item['avg_retransmits']}`{udp_text}{ping_text}"
            )
        lines.append("")
        for item in iperf_summary["by_as_status_and_hop"]:
            receiver_text = f", 平均 receiver `{item['avg_receiver_gbps']:.3f} Gbps`" if item["avg_receiver_gbps"] is not None else ""
            udp_text = (
                f", UDP 抖动 `{item['avg_udp_jitter_ms']:.6f} ms`, UDP 丢包 `{item['avg_udp_loss_pct']:.6f}%`, UDP 成功 `{item['udp_success_count']}/{item['count']}`"
                if item["avg_udp_jitter_ms"] is not None or item["avg_udp_loss_pct"] is not None
                else f", UDP 成功 `{item['udp_success_count']}/{item['count']}`"
            )
            ping_text = (
                f", ping 平均 RTT `{item['avg_ping_rtt_ms']:.6f} ms`, 最大 RTT `{item['max_ping_rtt_ms']:.6f} ms`"
                if item["avg_ping_rtt_ms"] is not None
                else ""
            )
            lines.append(
                f"- `{item['as_status']}` hop `{item['hop_count']}`: `{item['count']}` 条, sender 平均 `{item['avg_sender_gbps']:.3f} Gbps`, "
                f"最小 `{item['min_sender_gbps']:.3f}`, 最大 `{item['max_sender_gbps']:.3f}`, 平均重传 `{item['avg_retransmits']}`{receiver_text}{udp_text}{ping_text}"
            )
        lines.append("")
        best = iperf_summary["best_sender_run"]
        worst = iperf_summary["worst_sender_run"]
        lines.append(f"- 最佳 sender 带宽: `{best['file']}` -> `{best['sender_bitrate_gbps']:.3f} Gbps`, 重传 `{best['retransmits']}`")
        lines.append(f"- 最差 sender 带宽: `{worst['file']}` -> `{worst['sender_bitrate_gbps']:.3f} Gbps`, 重传 `{worst['retransmits']}`")
        if iperf_summary["best_ping_run"] is not None:
            lines.append(f"- 最低 ping 平均 RTT: `{iperf_summary['best_ping_run']['file']}` -> `{iperf_summary['best_ping_run']['ping_rtt_avg_ms']:.6f} ms`")
        if iperf_summary["worst_ping_run"] is not None:
            lines.append(f"- 最高 ping 平均 RTT: `{iperf_summary['worst_ping_run']['file']}` -> `{iperf_summary['worst_ping_run']['ping_rtt_avg_ms']:.6f} ms`")
        if iperf_summary["worst_udp_jitter_run"] is not None:
            lines.append(f"- 最高 UDP 抖动: `{iperf_summary['worst_udp_jitter_run']['file']}` -> `{iperf_summary['worst_udp_jitter_run']['udp_jitter_ms']:.6f} ms`")
        if iperf_summary["worst_udp_loss_run"] is not None:
            lines.append(f"- 最高 UDP 丢包率: `{iperf_summary['worst_udp_loss_run']['file']}` -> `{iperf_summary['worst_udp_loss_run']['udp_loss_pct']:.6f}%`")
    else:
        lines.append("- 未解析到 iperf 结果文件。")
    lines.append("")

    lines.append("## 6. 内存统计")
    lines.append("")
    if len(host_mem) >= 3:
        delta_0_5 = host_mem[1]["used_mem_mb"] - host_mem[0]["used_mem_mb"]
        delta_5_10 = host_mem[2]["used_mem_mb"] - host_mem[1]["used_mem_mb"]
        delta_0_10 = host_mem[2]["used_mem_mb"] - host_mem[0]["used_mem_mb"]
        lines.append(f"- 宿主机内存变化: step0->step5 `{delta_0_5:+.0f} MB`, step5->step10 `{delta_5_10:+.0f} MB`, step0->step10 `{delta_0_10:+.0f} MB`")
    for item in memory_snapshots:
        lines.append(
            f"- `{item['file']}`: Total_Mem_MB 平均 `{item['total_mem_mb_avg']:.4f}`, BIRD_Routing_Mem_MB 平均 `{item['bird_mem_mb_avg']:.4f}`, 容器数 `{item['containers']}`"
        )
    if memory_aggregate:
        lines.append(
            f"- hostMemory 全部快照平均: Total_Mem_MB `{memory_aggregate['avg_total_mem_mb_across_snapshots']:.4f}`, "
            f"BIRD_Routing_Mem_MB `{memory_aggregate['avg_bird_mem_mb_across_snapshots']:.4f}`"
        )
    lines.append("")

    lines.append("## 7. RTNL 与上下文切换")
    lines.append("")
    for entry in rtnl_entries:
        if entry.get("paired_rows", 0) == 0:
            continue
        wait_stats = entry["wait_us_stats"]
        hold_stats = entry["hold_us_stats"]
        total_stats = entry["total_lock_us_stats"]
        lines.append(
            f"- `{entry['file']}`: 原始 `{entry['raw_rows']}` 行, 成对聚合后 `{entry['paired_rows']}` 个操作, 平均 wait `{wait_stats['avg']}` us, "
            f"平均 hold `{hold_stats['avg']}` us, 平均总锁开销 `{total_stats['avg']}` us"
        )
        lines.append(
            f"  wait p50/p90/p99: `{wait_stats['median']}` / `{wait_stats['p90']}` / `{wait_stats['p99']}` us; "
            f"总锁开销 p50/p90/p99: `{total_stats['median']}` / `{total_stats['p90']}` / `{total_stats['p99']}` us"
        )
        lines.append(
            f"  最大 wait: `{entry['max_wait_entry']['wait_sum_us']}` us @ `{entry['max_wait_entry']['container']}`; "
            f"最大总锁开销: `{entry['max_total_lock_entry']['total_lock_us']}` us @ `{entry['max_total_lock_entry']['container']}`"
        )
        lines.append(f"  wait 分布: {', '.join(compress_distribution(entry['wait_distribution']))}")
        lines.append(f"  total 分布: {', '.join(compress_distribution(entry['total_lock_distribution']))}")
    for entry in csw_entries:
        if entry.get("rows", 0) == 0:
            continue
        lines.append(
            f"- `{entry['file']}`: `{entry['rows']}` 条 pidstat 记录, 平均 cswch/s `{entry['avg_cswch_s']}`, p95 `{entry['p95_cswch_s']}`, "
            f"峰值 `{entry['max_cswch_s']}` @ pid `{entry['max_cswch_pid']}`"
        )
    lines.append("")

    lines.append("## 8. lifecycle_trace 汇总")
    lines.append("")
    lines.append(f"- trace 总事件数: `{trace_aggregate['total_events']}`, 操作统计: `{json.dumps(trace_aggregate['counts_by_operation'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 9. 说明")
    lines.append("")
    lines.append("- MRT 的 UPDATE 峰值按 MRT 记录时间戳做按秒统计。")
    lines.append("- RTNL 统计先按相邻两行、相同 pid/container/inode、10ms 内配对，再将 wait 和 hold 分别求和。")
    lines.append("- iperf 分类优先使用 `iperf_tasks.json` 中的 `as_status` 和 `hop_count`，文件名中的 hop 只作为兜底。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment result logs.")
    parser.add_argument("result_dir", help="Path to result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    convergence_dir = result_dir / "convergence"
    seed_profiler_dir = result_dir / "seed_profiler"

    trace_logs = [parse_trace_log(path) for path in sorted(convergence_dir.glob("lifecycle_trace_*.log"))]
    memory_snapshots = parse_memory_snapshots(result_dir / "hostMemory")
    iperf_data = parse_iperf_results(result_dir)

    data = {
        "result_dir": result_dir,
        "docker_build": parse_docker_build(next((result_dir / "dockerBuild").glob("dockerBuild_*.log"))),
        "docker_up": parse_docker_up(next((result_dir / "dockerBuild").glob("dockerUp_*.log"))),
        "host_mem": parse_host_mem_csvs(result_dir / "memory"),
        "memory_snapshots": memory_snapshots,
        "memory_aggregate": aggregate_memory_snapshots(memory_snapshots),
        "phases": parse_all_phase_summaries(convergence_dir),
        "trace_logs": trace_logs,
        "trace_aggregate": aggregate_trace_logs(trace_logs),
        "rtnl": [parse_rtnl_log(path) for path in sorted(seed_profiler_dir.glob("rtnl_trace_*.log"))],
        "csw": [parse_csw_log(path) for path in sorted(seed_profiler_dir.glob("csw_trace_*.log"))],
        "mrt_files": parse_mrt_files(result_dir / "mrt"),
        "iperf": iperf_data,
        "iperf_summary": summarize_iperf(iperf_data["entries"]),
    }
    data["phase_summary"] = summarize_phases(data["phases"])
    data["timeline"] = build_timeline(data["host_mem"], data["docker_up"], data["phases"])

    md_path = result_dir / "log_analysis_report.md"
    json_path = result_dir / "log_analysis_report.json"
    md_path.write_text(generate_report(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

    print(f"Report written: {md_path}")
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
