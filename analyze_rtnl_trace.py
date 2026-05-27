import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median


TIME_FMT_MS = "%Y-%m-%d %H:%M:%S.%f"
RTNL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<comm>\S+)\s+"
    r"(?P<op>\S+)\s+"
    r"(?P<container>.+?)\s+\((?P<inode>\d+)\)\s+"
    r"(?P<wait>\d+)\s+(?P<hold>\d+)\s*$"
)


def safe_mean(values):
    return mean(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return float(lower_value + (upper_value - lower_value) * (pos - lower))


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


def parse_rtnl_rows(path):
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = RTNL_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "raw_line": line,
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


def merge_rtnl_rows_by_adjacent_pid(rows):
    merged = []
    idx = 0
    while idx < len(rows):
        current = rows[idx]
        group_rows = [current]
        idx += 1
        while idx < len(rows) and rows[idx]["pid"] == current["pid"]:
            group_rows.append(rows[idx])
            idx += 1

        merged.append(
            {
                "ts": group_rows[0]["ts"],
                "pid": group_rows[0]["pid"],
                "comm": group_rows[0]["comm"],
                "container": group_rows[0]["container"],
                "inode": group_rows[0]["inode"],
                "ops": [row["op"] for row in group_rows],
                "group_size": len(group_rows),
                "containers": sorted({row["container"] for row in group_rows}),
                "inodes": sorted({row["inode"] for row in group_rows}),
                "wait_sum_us": sum(row["wait_us"] for row in group_rows),
                "hold_sum_us": sum(row["hold_us"] for row in group_rows),
                "total_lock_us": sum(row["wait_us"] + row["hold_us"] for row in group_rows),
                "source_lines": [row["raw_line"] for row in group_rows],
            }
        )
    return merged


def build_rtnl_summary(entries, label):
    if not entries:
        return {
            "file": label,
            "raw_rows": 0,
            "merged_rows": 0,
            "wait_us_stats": {},
            "hold_us_stats": {},
            "total_lock_us_stats": {},
            "wait_distribution": "",
            "hold_distribution": "",
            "total_lock_distribution": "",
        }

    waits = [row["wait_sum_us"] for row in entries]
    holds = [row["hold_sum_us"] for row in entries]
    totals = [row["total_lock_us"] for row in entries]

    return {
        "file": label,
        "raw_rows": sum(len(row["source_lines"]) for row in entries),
        "merged_rows": len(entries),
        "wait_us_stats": summarize_numeric(waits),
        "hold_us_stats": summarize_numeric(holds),
        "total_lock_us_stats": summarize_numeric(totals),
        "wait_distribution": ", ".join(compress_distribution(build_log2_distribution(waits))),
        "hold_distribution": ", ".join(compress_distribution(build_log2_distribution(holds))),
        "total_lock_distribution": ", ".join(
            compress_distribution(build_log2_distribution(totals))
        ),
    }


def parse_rtnl_log(path):
    raw_rows = parse_rtnl_rows(path)
    if not raw_rows:
        return {"file": path.name, "raw_rows": 0, "merged_rows": 0}
    merged_rows = merge_rtnl_rows_by_adjacent_pid(raw_rows)
    return build_rtnl_summary(merged_rows, path.name)


def iter_rtnl_logs(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        seed_profiler_dir = path / "seed_profiler"
        search_dir = seed_profiler_dir if seed_profiler_dir.is_dir() else path
        return sorted(search_dir.rglob("rtnl_trace*.log"))
    raise FileNotFoundError(f"Path not found: {input_path}")


def resolve_output_dir(input_path):
    path = Path(input_path)
    if path.is_file():
        return path.parent
    if path.is_dir():
        seed_profiler_dir = path / "seed_profiler"
        return seed_profiler_dir if seed_profiler_dir.is_dir() else path
    raise FileNotFoundError(f"Path not found: {input_path}")


def build_output_payload(summary):
    return {
        "file": summary["file"],
        "raw_rows": summary["raw_rows"],
        "merged_rows": summary["merged_rows"],
        "wait_us_stats": summary["wait_us_stats"],
        "hold_us_stats": summary["hold_us_stats"],
        "total_lock_us_stats": summary["total_lock_us_stats"],
        "wait_distribution": summary["wait_distribution"],
        "hold_distribution": summary["hold_distribution"],
        "total_lock_distribution": summary["total_lock_distribution"],
    }


def write_summary_files(output_dir, summary, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_output_payload(summary)
    json_path = output_dir / f"{stem}.json"

    #json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"file: {payload['file']}",
        f"raw_rows: {payload['raw_rows']}",
        f"merged_rows: {payload['merged_rows']}",
        f"wait_us_stats: {payload['wait_us_stats']}",
        f"hold_us_stats: {payload['hold_us_stats']}",
        f"total_lock_us_stats: {payload['total_lock_us_stats']}",
        f"wait_distribution: {payload['wait_distribution']}",
        f"hold_distribution: {payload['hold_distribution']}",
        f"total_lock_distribution: {payload['total_lock_distribution']}",
    ]
    return json_path


def print_text_summary(summary):
    print(f"file: {summary['file']}")
    print(f"raw_rows: {summary['raw_rows']}")
    print(f"merged_rows: {summary['merged_rows']}")
    if summary["merged_rows"] == 0:
        return

    print(f"wait_us_stats: {summary['wait_us_stats']}")
    print(f"hold_us_stats: {summary['hold_us_stats']}")
    print(f"total_lock_us_stats: {summary['total_lock_us_stats']}")
    print(f"wait_distribution: {summary['wait_distribution']}")
    print(f"hold_distribution: {summary['hold_distribution']}")
    print(f"total_lock_distribution: {summary['total_lock_distribution']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze rtnl_trace*.log files.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=".",
        help="A single rtnl log file or a directory containing rtnl_trace*.log files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the aggregated result to this JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.input_path)
    log_paths = iter_rtnl_logs(args.input_path)
    summaries = [parse_rtnl_log(path) for path in log_paths]
    aggregated_entries = []
    for path in log_paths:
        aggregated_entries.extend(merge_rtnl_rows_by_adjacent_pid(parse_rtnl_rows(path)))
    aggregate_summary = build_rtnl_summary(aggregated_entries, "ALL_RTNL_TRACE_FILES")

    for summary in summaries:
        print_text_summary(summary)
        print()
        stem = Path(summary["file"]).stem + "_analysis"
        json_path= write_summary_files(output_dir, summary, stem)
        #print(f"saved: {json_path}")
        print()

    if len(summaries) > 1:
        print_text_summary(aggregate_summary)
        print()
        json_path = write_summary_files(output_dir, aggregate_summary, "rtnl_trace_aggregate")
        #print(f"saved: {json_path}")
        print()

    if args.output_json:
        output_path = Path(args.output_json)
        payload = {
            "files": [build_output_payload(summary) for summary in summaries],
            "aggregate": build_output_payload(aggregate_summary),
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json saved to {output_path}")


if __name__ == "__main__":
    main()
