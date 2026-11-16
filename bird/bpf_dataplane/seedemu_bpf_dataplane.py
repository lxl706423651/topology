#!/usr/bin/env python3
"""Install a TC/BPF route-map dataplane for running SeedEMU brdnode containers.

This is an experiment line separate from F/K1/K2c/K3c. It does not write remote
routes into Linux main FIB. Instead, it reads selected BIRD routes and updates a
per-router BPF LPM trie. The TC program forwards packets before Linux FIB lookup
when a BPF route exists; misses fall back to the normal Linux stack.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR / "tc_lpm_router.bpf.c"
BUILD_DIR = SCRIPT_DIR / "build"
OBJ = BUILD_DIR / "tc_lpm_router.bpf.o"
MODE_FILE = BUILD_DIR / "tc_lpm_router.bpf.mode"
APPLY_HELPER_SRC = SCRIPT_DIR / "bpf_map_apply.c"
APPLY_HELPER = BUILD_DIR / "bpf_map_apply"
DEFAULT_PIN_ROOT = Path("/sys/fs/bpf/seedemu_bpf")
DEFAULT_RUN_ROOT = Path("/home/lxl/bird/runs")

BRDNODE_RE = re.compile(r"(?:^|[-_])(?:as(\d+)brd-r(\d+)|brdnode_(\d+)_r(\d+))(?:$|[-_])")
IPV4_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\s+unicast\b.*\*")
ROUTE_PROTO_RE = re.compile(r"\[([^\s\]]+)")
VIA_RE = re.compile(r"^\s+via\s+(\d{1,3}(?:\.\d{1,3}){3})\s+on\s+(\S+)")
DEV_RE = re.compile(r"^\s+dev\s+(\S+)")
BATCH_UPDATE_RE = re.compile(r"^map update pinned \S+ key hex (.+?) value hex (.+)$")
# bpffs pin paths are stricter than ordinary filesystem paths on some kernels.
# Keep project/container-derived names conservative to avoid EPERM on mkdir/pin.
SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")
BPFT = "bpftool"
DEFAULT_SKIP_PREFIXES = ["10.0.0.0/8"]


def run(
    cmd: list[str],
    *,
    check: bool = False,
    timeout: float = 120,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_text, text=True, capture_output=True, timeout=timeout, check=check)


def need_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("ERROR: run as root on the VM, e.g. sudo python3 seedemu_bpf_dataplane.py ...")


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"ERROR: required tool not found: {name}")
    return path


def usable_bpftool() -> str:
    candidates: list[str] = []
    wrapper = shutil.which("bpftool")
    if wrapper:
        candidates.append(wrapper)
    candidates.extend(str(p) for p in sorted(Path("/usr/lib/linux-tools").glob("*/bpftool"), reverse=True))
    for candidate in candidates:
        proc = run([candidate, "version"], timeout=10)
        if proc.returncode == 0:
            return candidate
    raise SystemExit("ERROR: no usable bpftool found")


def clang_bin() -> str:
    for name in ("clang", "clang-18", "clang-17", "clang-16"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("ERROR: clang not found; install clang/llvm first")


def compile_bpf(force: bool = False, decrement_ttl: bool = False) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    mode = "decrement_ttl=1\n" if decrement_ttl else "decrement_ttl=0\n"
    if (
        OBJ.exists()
        and MODE_FILE.exists()
        and MODE_FILE.read_text(encoding="utf-8") == mode
        and not force
        and OBJ.stat().st_mtime >= SRC.stat().st_mtime
    ):
        return
    clang = clang_bin()
    defines = ["-DSEED_BPF_DECREMENT_TTL"] if decrement_ttl else []
    proc = run(
        [
            clang,
            "-O2",
            "-g",
            "-target",
            "bpf",
            "-D__TARGET_ARCH_x86",
            *defines,
            "-Wall",
            "-Werror",
            "-I/usr/include",
            "-I/usr/include/x86_64-linux-gnu",
            "-c",
            str(SRC),
            "-o",
            str(OBJ),
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(((proc.stdout or "") + (proc.stderr or ""))[-4000:])
    MODE_FILE.write_text(mode, encoding="utf-8")


def compile_apply_helper(force: bool = False) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if APPLY_HELPER.exists() and not force and APPLY_HELPER.stat().st_mtime >= APPLY_HELPER_SRC.stat().st_mtime:
        return
    cc = None
    for name in ("gcc", "cc", "clang"):
        cc = shutil.which(name)
        if cc:
            break
    if not cc:
        raise SystemExit("ERROR: gcc/cc/clang not found; install a C compiler first")
    proc = run(
        [
            cc,
            "-O2",
            "-Wall",
            "-Wextra",
            "-D_GNU_SOURCE",
            str(APPLY_HELPER_SRC),
            "-o",
            str(APPLY_HELPER),
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(((proc.stdout or "") + (proc.stderr or ""))[-4000:])


def safe_name(name: str) -> str:
    return SAFE_RE.sub("_", name).strip("_") or "container"


def brdnode_match(name: str) -> tuple[int, int] | None:
    match = BRDNODE_RE.search(name)
    if not match:
        return None
    if match.group(1) is not None:
        return int(match.group(1)), int(match.group(2))
    return int(match.group(3)), int(match.group(4))


def docker_brdnodes(project: str, limit: int = 0) -> list[dict]:
    ps = run(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}}"], check=True)
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        return []
    inspect = run(["docker", "inspect", "-f", "{{.Id}} {{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids, check=True)
    nodes = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cid, name, pid_text, running = parts[:4]
        name = name.lstrip("/")
        parsed = brdnode_match(name)
        if running.lower() != "true" or not parsed:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid <= 0:
            continue
        asn, rnum = parsed
        nodes.append({"id": cid, "name": name, "pid": pid, "as": asn, "r": rnum})
    nodes.sort(key=lambda item: (item["as"], item["r"], item["name"]))
    if limit > 0:
        nodes = nodes[:limit]
    return nodes


def parse_bird_primary_routes(text: str) -> tuple[list[dict], dict]:
    routes = []
    current: dict | None = None
    stats = {
        "prefixes_seen": 0,
        "routes_with_nexthop": 0,
        "via_routes": 0,
        "dev_only_routes": 0,
        "multipath_routes": 0,
    }
    for line in text.splitlines():
        route_match = IPV4_PREFIX_RE.match(line)
        if route_match:
            stats["prefixes_seen"] += 1
            if current and current["nexthops"]:
                routes.append(current)
            proto_match = ROUTE_PROTO_RE.search(line)
            current = {
                "prefix": route_match.group(1),
                "protocol": proto_match.group(1) if proto_match else "",
                "nexthops": [],
            }
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

    stats["routes_with_nexthop"] = len(routes)
    for route in routes:
        hops = route["nexthops"]
        if len(hops) > 1:
            stats["multipath_routes"] += 1
        hop = hops[0]
        if hop.get("via"):
            stats["via_routes"] += 1
        else:
            stats["dev_only_routes"] += 1
    return routes, stats


def container_interfaces(name: str, timeout: float) -> tuple[list[str], dict[str, int]]:
    script = "for d in /sys/class/net/*; do n=${d##*/}; printf '%s %s\\n' \"$n\" \"$(cat \"$d/ifindex\")\"; done"
    proc = run(["docker", "exec", name, "sh", "-lc", script], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
    ifindices: dict[str, int] = {}
    interfaces = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        dev, idx = parts
        try:
            ifindex = int(idx)
        except ValueError:
            continue
        ifindices[dev] = ifindex
        if dev != "lo":
            interfaces.append(dev)
    return interfaces, ifindices


def disable_rp_filter(name: str, timeout: float) -> int:
    script = "for f in /proc/sys/net/ipv4/conf/*/rp_filter; do echo 0 > \"$f\" 2>/dev/null || true; done"
    proc = run(["docker", "exec", name, "sh", "-lc", script], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
    verify = run(
        ["docker", "exec", name, "sh", "-lc", "grep -L '^0$' /proc/sys/net/ipv4/conf/*/rp_filter | wc -l"],
        timeout=timeout,
    )
    if verify.returncode != 0:
        raise RuntimeError((verify.stderr or verify.stdout)[-2000:])
    return int((verify.stdout or "0").strip() or "0")


def dump_bird_routes(name: str, timeout: float) -> tuple[list[dict], dict]:
    proc = run(
        ["docker", "exec", name, "birdc", "-s", "/run/bird/bird.ctl", "show", "route", "primary"],
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "birdc show route primary failed")[-4000:])
    return parse_bird_primary_routes(proc.stdout)


def pin_dir(pin_root: Path, project: str, node_name: str) -> Path:
    return pin_root / safe_name(project) / safe_name(node_name)


def prog_path(pin_root: Path, project: str, node_name: str) -> Path:
    return pin_dir(pin_root, project, node_name) / "tc_router"


def map_path(pin_root: Path, project: str, node_name: str, map_name: str) -> Path:
    return pin_dir(pin_root, project, node_name) / "maps" / map_name


def hex_bytes(buf: bytes) -> list[str]:
    return [f"{b:02x}" for b in buf]


def route_update_command(map_file: Path, prefix: str, ifindex: int, nexthop: str) -> str:
    key = prefix_to_key(prefix)
    value = route_value(ifindex, nexthop)
    return (
        f"map update pinned {map_file} "
        f"key hex {' '.join(hex_bytes(key))} "
        f"value hex {' '.join(hex_bytes(value))}"
    )


def route_delete_command(map_file: Path, key_hex: str) -> str:
    return f"map delete pinned {map_file} key hex {key_hex}"


def prefix_to_key(prefix: str) -> bytes:
    addr_text, prefixlen_text = prefix.split("/", 1)
    return struct.pack("<I", int(prefixlen_text)) + socket.inet_aton(addr_text)


def route_value(ifindex: int, nexthop: str) -> bytes:
    return struct.pack("<I", ifindex) + socket.inet_aton(nexthop)


def hex_text(buf: bytes) -> str:
    return " ".join(hex_bytes(buf))


def parse_skip_prefix(text: str) -> tuple[int, int, int]:
    addr_text, prefixlen_text = text.split("/", 1)
    prefixlen = int(prefixlen_text)
    addr = struct.unpack("!I", socket.inet_aton(addr_text))[0]
    mask = (0xFFFFFFFF << (32 - prefixlen)) & 0xFFFFFFFF if prefixlen else 0
    return addr & mask, prefixlen, mask


def prefix_is_skipped(prefix: str, skip_prefixes: list[tuple[int, int, int]]) -> bool:
    if not skip_prefixes:
        return False
    addr_text, prefixlen_text = prefix.split("/", 1)
    prefixlen = int(prefixlen_text)
    addr = struct.unpack("!I", socket.inet_aton(addr_text))[0]
    for skip_addr, skip_len, skip_mask in skip_prefixes:
        if prefixlen >= skip_len and (addr & skip_mask) == skip_addr:
            return True
    return False


def parse_update_commands(commands: list[str]) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for command in commands:
        match = BATCH_UPDATE_RE.match(command.strip())
        if not match:
            continue
        key_hex = " ".join(match.group(1).split())
        value_hex = " ".join(match.group(2).split())
        parsed[key_hex] = (value_hex, command.strip())
    return parsed


def read_route_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries", data)
    if not isinstance(entries, dict):
        return {}
    return {str(key): str(value) for key, value in entries.items()}


def write_route_cache(path: Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, separators=(",", ":")) + "\n", encoding="utf-8")


def commands_from_entries(entries: dict[str, str], map_file: Path) -> list[str]:
    return [
        f"map update pinned {map_file} key hex {key} value hex {value}"
        for key, value in sorted(entries.items())
    ]


def delta_from_entry_maps(
    old: dict[str, str],
    new: dict[str, str],
    map_file: Path,
) -> tuple[list[tuple[str, str | None]], dict[str, int]]:
    deletes = [(key, None) for key in sorted(old.keys() - new.keys())]
    updates = [
        (key, new[key])
        for key in sorted(new)
        if old.get(key) != new[key]
    ]
    return deletes + updates, {
        "route_ops_applied": len(deletes) + len(updates),
        "route_deletes_applied": len(deletes),
        "route_updates_applied": len(updates),
        "route_delta_old_keys": len(old),
        "route_delta_new_keys": len(new),
    }


def delta_apply_commands(
    old_commands: list[str],
    new_commands: list[str],
    map_file: Path,
) -> tuple[list[str], dict[str, int]]:
    old = parse_update_commands(old_commands)
    new = parse_update_commands(new_commands)
    deletes = [
        route_delete_command(map_file, key)
        for key in sorted(old.keys() - new.keys())
    ]
    updates = [
        new[key][1]
        for key in sorted(new)
        if old.get(key, (None, ""))[0] != new[key][0]
    ]
    return deletes + updates, {
        "route_ops_applied": len(deletes) + len(updates),
        "route_deletes_applied": len(deletes),
        "route_updates_applied": len(updates),
        "route_delta_old_keys": len(old),
        "route_delta_new_keys": len(new),
    }


def write_binary_ops(path: Path, ops: list[tuple[str, str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for key_hex, value_hex in ops:
            key = bytes.fromhex(key_hex)
            if value_hex is None:
                value = b"\x00" * 8
                op = 2
            else:
                value = bytes.fromhex(value_hex)
                op = 1
            if len(key) != 8 or len(value) != 8:
                raise ValueError(f"unexpected key/value length for BPF route op: {len(key)}/{len(value)}")
            fh.write(struct.pack("<B3x", op))
            fh.write(key)
            fh.write(value)


def run_apply_helper(map_file: Path, ops_file: Path, timeout: float) -> None:
    proc = run([str(APPLY_HELPER), str(map_file), str(ops_file)], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(((proc.stdout or "") + (proc.stderr or ""))[-4000:])


def load_program(pin_root: Path, project: str, node: dict) -> None:
    pdir = pin_dir(pin_root, project, node["name"])
    run(["rm", "-rf", str(pdir)], check=False)
    (pdir / "maps").mkdir(parents=True, exist_ok=True)
    proc = run([BPFT, "prog", "loadall", str(OBJ), str(pdir), "pinmaps", str(pdir / "maps")], timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])


def attach_program(pin_root: Path, project: str, node: dict, interfaces: list[str], timeout: float) -> list[str]:
    attached = []
    net_arg = f"--net=/proc/{node['pid']}/ns/net"
    ppath = str(prog_path(pin_root, project, node["name"]))
    for dev in interfaces:
        qdisc = run(["nsenter", net_arg, "--", "tc", "qdisc", "replace", "dev", dev, "clsact"], timeout=timeout)
        if qdisc.returncode != 0:
            raise RuntimeError(f"tc qdisc {dev}: {(qdisc.stderr or qdisc.stdout)[-1000:]}")
        filt = run(
            ["nsenter", net_arg, "--", "tc", "filter", "replace", "dev", dev, "ingress", "pref", "10", "bpf", "da", "pinned", ppath],
            timeout=timeout,
        )
        if filt.returncode != 0:
            raise RuntimeError(f"tc filter {dev}: {(filt.stderr or filt.stdout)[-1000:]}")
        attached.append(dev)
    return attached


def cleanup_node(pin_root: Path, project: str, node: dict, timeout: float) -> dict:
    result = {"name": node["name"], "tc_deleted": 0, "errors": []}
    try:
        interfaces, _ = container_interfaces(node["name"], timeout)
    except Exception as exc:
        interfaces = []
        result["errors"].append(str(exc))
    net_arg = f"--net=/proc/{node['pid']}/ns/net"
    for dev in interfaces:
        proc = run(["nsenter", net_arg, "--", "tc", "qdisc", "del", "dev", dev, "clsact"], timeout=timeout)
        if proc.returncode == 0:
            result["tc_deleted"] += 1
    run(["rm", "-rf", str(pin_dir(pin_root, project, node["name"]))], check=False)
    return result


def load_interface_cache(run_dir: Path | None) -> dict[str, tuple[list[str], dict[str, int]]]:
    if not run_dir:
        return {}
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cache: dict[str, tuple[list[str], dict[str, int]]] = {}
    for row in summary.get("rows") or []:
        name = row.get("name")
        interfaces = row.get("interfaces")
        ifindices = row.get("ifindices")
        if not name or not isinstance(interfaces, list) or not isinstance(ifindices, dict):
            continue
        try:
            cache[str(name)] = ([str(item) for item in interfaces], {str(k): int(v) for k, v in ifindices.items()})
        except Exception:
            continue
    return cache


def install_one(node: dict, args: argparse.Namespace, run_dir: Path) -> dict:
    started = time.monotonic()
    last_phase = started
    phase_seconds: dict[str, float] = {}

    def mark_phase(name: str) -> None:
        nonlocal last_phase
        now = time.monotonic()
        phase_seconds[name] = now - last_phase
        last_phase = now

    result = {
        "name": node["name"],
        "id": node["id"][:12],
        "pid": node["pid"],
        "as": node["as"],
        "r": node["r"],
        "success": False,
        "interfaces": [],
        "routes_installed": 0,
    }
    try:
        cached_interfaces = args.interface_cache.get(node["name"]) if getattr(args, "interface_cache", None) else None
        if cached_interfaces:
            interfaces, ifindices = cached_interfaces
            interface_cache_hit = True
        else:
            interfaces, ifindices = container_interfaces(node["name"], args.command_timeout)
            interface_cache_hit = False
        mark_phase("interfaces")
        if args.skip_rp_filter_disable:
            rp_filter_nonzero = -1
        else:
            rp_filter_nonzero = disable_rp_filter(node["name"], args.command_timeout)
        mark_phase("disable_rp_filter")
        routes, route_stats = dump_bird_routes(node["name"], args.birdc_timeout)
        mark_phase("dump_bird_routes")

        entries: dict[str, str] = {}
        skipped_protocol = 0
        skipped_prefix = 0
        skipped_missing_ifindex = 0
        skipped_no_via = 0
        skipped_multipath = 0
        for route in routes:
            protocol = route.get("protocol") or ""
            if args.include_protocol_regex and not args.include_protocol_regex.search(protocol):
                skipped_protocol += 1
                continue
            if prefix_is_skipped(route["prefix"], args.skip_prefixes):
                skipped_prefix += 1
                continue
            hops = route.get("nexthops") or []
            if not hops:
                skipped_no_via += 1
                continue
            if len(hops) > 1:
                if args.multipath_policy == "skip":
                    skipped_multipath += 1
                    continue
            hop = hops[0]
            via = hop.get("via")
            dev = hop.get("dev")
            if not via or not dev:
                skipped_no_via += 1
                continue
            ifindex = ifindices.get(dev)
            if ifindex is None:
                skipped_missing_ifindex += 1
                continue
            entries[hex_text(prefix_to_key(route["prefix"]))] = hex_text(route_value(ifindex, via))
        mark_phase("build_commands")

        map_file = map_path(args.pin_root, args.project_name, node["name"], "routes")
        commands: list[str] | None = None

        def get_commands() -> list[str]:
            nonlocal commands
            if commands is None:
                commands = commands_from_entries(entries, map_file)
            return commands

        batch_file = run_dir / "bpftool_batches" / f"{safe_name(node['name'])}.bpftool"
        if args.write_text_batches:
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            text_commands = get_commands()
            batch_file.write_text("\n".join(text_commands) + ("\n" if text_commands else ""), encoding="utf-8")

        cache_file = run_dir / "route_cache" / f"{safe_name(node['name'])}.json"
        write_route_cache(cache_file, entries)

        need_load_attach = True
        apply_ops: list[tuple[str, str | None]] = [(key, value) for key, value in sorted(entries.items())]
        apply_commands: list[str] | None = commands
        delta_stats = {
            "route_ops_applied": len(apply_ops),
            "route_deletes_applied": 0,
            "route_updates_applied": len(apply_ops),
            "route_delta_old_keys": 0,
            "route_delta_new_keys": len(entries),
            "route_delta_mode": False,
            "route_delta_baseline_missing": False,
            "route_delta_cache_hit": False,
        }
        if args.delta_from_run_dir and map_file.exists():
            previous_batch = args.delta_from_run_dir / "bpftool_batches" / f"{safe_name(node['name'])}.bpftool"
            previous_cache = args.delta_from_run_dir / "route_cache" / f"{safe_name(node['name'])}.json"
            old_entries = read_route_cache(previous_cache)
            if old_entries:
                apply_ops, delta_stats = delta_from_entry_maps(old_entries, entries, map_file)
                if args.write_text_batches or args.apply_mode == "bpftool":
                    apply_commands = [
                        route_delete_command(map_file, key) if value is None
                        else f"map update pinned {map_file} key hex {key} value hex {value}"
                        for key, value in apply_ops
                    ]
                delta_stats["route_delta_mode"] = True
                delta_stats["route_delta_baseline_missing"] = False
                delta_stats["route_delta_cache_hit"] = True
                need_load_attach = False
            elif previous_batch.exists():
                old_commands = previous_batch.read_text(encoding="utf-8").splitlines()
                apply_commands, delta_stats = delta_apply_commands(old_commands, get_commands(), map_file)
                apply_ops = []
                for command in apply_commands:
                    if command.startswith("map delete pinned "):
                        apply_ops.append((" ".join(command.split(" key hex ", 1)[1].split()), None))
                    else:
                        match = BATCH_UPDATE_RE.match(command.strip())
                        if match:
                            apply_ops.append((" ".join(match.group(1).split()), " ".join(match.group(2).split())))
                delta_stats["route_delta_mode"] = True
                delta_stats["route_delta_baseline_missing"] = False
                delta_stats["route_delta_cache_hit"] = False
                need_load_attach = False
            else:
                delta_stats["route_delta_baseline_missing"] = True
        mark_phase("delta_compare")

        if need_load_attach:
            load_program(args.pin_root, args.project_name, node)
        mark_phase("load_program")

        apply_batch_file = batch_file
        apply_binary_file = run_dir / "bpf_ops" / f"{safe_name(node['name'])}.bin"
        if args.delta_from_run_dir:
            apply_batch_file = run_dir / "bpftool_delta_batches" / f"{safe_name(node['name'])}.bpftool"
            if args.write_text_batches:
                apply_batch_file.parent.mkdir(parents=True, exist_ok=True)
                text_commands = apply_commands or []
                apply_batch_file.write_text("\n".join(text_commands) + ("\n" if text_commands else ""), encoding="utf-8")

        if apply_ops:
            if args.apply_mode == "syscall":
                write_binary_ops(apply_binary_file, apply_ops)
                run_apply_helper(map_file, apply_binary_file, args.command_timeout)
            else:
                if not args.write_text_batches:
                    text_commands = apply_commands if apply_commands is not None else get_commands()
                    apply_batch_file.parent.mkdir(parents=True, exist_ok=True)
                    apply_batch_file.write_text("\n".join(text_commands) + ("\n" if text_commands else ""), encoding="utf-8")
                proc = run([BPFT, "batch", "file", str(apply_batch_file)], timeout=args.command_timeout)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stdout[-4000:])
        mark_phase("batch_update")

        attached = []
        if need_load_attach:
            attached = attach_program(args.pin_root, args.project_name, node, interfaces, args.command_timeout)
        mark_phase("attach_program")

        result.update(route_stats)
        result.update(
            {
                "interfaces": interfaces,
                "ifindices": ifindices,
                "interface_cache_hit": interface_cache_hit,
                "interfaces_attached": attached,
                "rp_filter_nonzero_after_disable": rp_filter_nonzero,
                "routes_installed": len(entries),
                "skipped_protocol": skipped_protocol,
                "skipped_prefix": skipped_prefix,
                "skipped_missing_ifindex": skipped_missing_ifindex,
                "skipped_no_via": skipped_no_via,
                "skipped_multipath": skipped_multipath,
                "batch_file": str(batch_file),
                "apply_batch_file": str(apply_batch_file),
                "apply_binary_file": str(apply_binary_file),
                "phase_seconds": phase_seconds,
                "success": True,
                **delta_stats,
            }
        )
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc(limit=8)
    finally:
        result["duration_seconds"] = time.monotonic() - started
    return result


def aggregate(
    rows: list[dict],
    started: float,
    wall_started_at: dt.datetime,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    ok = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]
    phase_totals: dict[str, float] = {}
    phase_max: dict[str, float] = {}
    for row in ok:
        for name, value in (row.get("phase_seconds") or {}).items():
            number = float(value)
            phase_totals[name] = phase_totals.get(name, 0.0) + number
            phase_max[name] = max(phase_max.get(name, 0.0), number)
    phase_avg = {
        name: total / len(ok)
        for name, total in phase_totals.items()
    } if ok else {}
    return {
        "schema_version": 1,
        "started_at": wall_started_at.isoformat(),
        "ended_at": dt.datetime.now().isoformat(),
        "duration_seconds": time.monotonic() - started,
        "project_name": args.project_name,
        "run_dir": str(run_dir),
        "pin_root": str(args.pin_root / safe_name(args.project_name)),
        "bpftool": BPFT,
        "apply_mode": args.apply_mode,
        "write_text_batches": args.write_text_batches,
        "object": str(OBJ),
        "apply_helper": str(APPLY_HELPER) if args.apply_mode == "syscall" else "",
        "nodes_total": len(rows),
        "nodes_success": len(ok),
        "nodes_failed": len(failed),
        "success": not failed and bool(rows),
        "routes_installed": sum(int(row.get("routes_installed") or 0) for row in ok),
        "route_ops_applied": sum(int(row.get("route_ops_applied") or 0) for row in ok),
        "route_deletes_applied": sum(int(row.get("route_deletes_applied") or 0) for row in ok),
        "route_updates_applied": sum(int(row.get("route_updates_applied") or 0) for row in ok),
        "route_delta_baseline_missing": sum(1 for row in ok if row.get("route_delta_baseline_missing")),
        "route_delta_cache_hits": sum(1 for row in ok if row.get("route_delta_cache_hit")),
        "interface_cache_hits": sum(1 for row in ok if row.get("interface_cache_hit")),
        "prefixes_seen": sum(int(row.get("prefixes_seen") or 0) for row in ok),
        "via_routes": sum(int(row.get("via_routes") or 0) for row in ok),
        "dev_only_routes": sum(int(row.get("dev_only_routes") or 0) for row in ok),
        "multipath_routes": sum(int(row.get("multipath_routes") or 0) for row in ok),
        "phase_seconds_total": phase_totals,
        "phase_seconds_avg": phase_avg,
        "phase_seconds_max": phase_max,
        "failed_nodes": [{"name": row.get("name"), "error": row.get("error")} for row in failed[:50]],
        "rows": rows,
    }


def main() -> int:
    global BPFT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--pin-root", type=Path, default=DEFAULT_PIN_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit-containers", type=int, default=0)
    parser.add_argument("--birdc-timeout", type=float, default=180)
    parser.add_argument("--command-timeout", type=float, default=180)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--apply-mode",
        choices=["bpftool", "syscall"],
        default="bpftool",
        help="Use bpftool batch files or the direct bpf(2) syscall helper to apply map updates.",
    )
    parser.add_argument(
        "--no-text-batches",
        dest="write_text_batches",
        action="store_false",
        help="Do not write bpftool text batch files. Route cache JSON and binary op files are still written as needed.",
    )
    parser.set_defaults(write_text_batches=True)
    parser.add_argument(
        "--skip-rp-filter-disable",
        action="store_true",
        help="Skip rp_filter writes. Use after a previous install already disabled rp_filter in all brdnodes.",
    )
    parser.add_argument("--multipath-policy", choices=["first", "skip"], default="first")
    parser.add_argument(
        "--delta-from-run-dir",
        type=Path,
        default=None,
        help=(
            "Read previous bpftool_batches from this run directory and apply only changed "
            "BPF map keys to already-attached routers. Falls back to full load/attach when "
            "a baseline batch or pinned map is missing."
        ),
    )
    parser.add_argument(
        "--interface-cache-run-dir",
        type=Path,
        default=None,
        help=(
            "Reuse interfaces/ifindex data from a previous BPF run summary. "
            "Defaults to --delta-from-run-dir when present."
        ),
    )
    parser.add_argument(
        "--decrement-ttl",
        action="store_true",
        help="Decrement IPv4 TTL in the TC/BPF program. Disabled by default for the current Docker/veth prototype.",
    )
    parser.add_argument(
        "--include-protocol-regex",
        default="^(Ebgp|Ibgp)",
        help="Only install routes whose BIRD source protocol matches this regex. Empty string disables the filter.",
    )
    parser.add_argument(
        "--skip-prefix",
        action="append",
        default=[],
        help="Do not install routes inside this prefix. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-skip-prefixes",
        action="store_true",
        help="Do not skip the default SeedEMU internal/control prefix 10.0.0.0/8.",
    )
    args = parser.parse_args()

    need_root()
    for name in ("docker", "nsenter", "tc", "ip", "rm"):
        tool(name)
    BPFT = usable_bpftool()
    compile_bpf(force=args.force_compile, decrement_ttl=args.decrement_ttl)
    if args.apply_mode == "syscall":
        compile_apply_helper(force=args.force_compile)
    args.include_protocol_regex = (
        re.compile(args.include_protocol_regex) if args.include_protocol_regex else None
    )
    skip_prefix_texts = ([] if args.no_default_skip_prefixes else DEFAULT_SKIP_PREFIXES) + args.skip_prefix
    args.skip_prefixes = [parse_skip_prefix(text) for text in skip_prefix_texts]
    if args.interface_cache_run_dir is None and args.delta_from_run_dir is not None:
        args.interface_cache_run_dir = args.delta_from_run_dir
    args.interface_cache = load_interface_cache(args.interface_cache_run_dir)

    started = time.monotonic()
    wall_started_at = dt.datetime.now()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (DEFAULT_RUN_ROOT / f"bpf_dataplane_{safe_name(args.project_name)}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or (run_dir / "summary.json")

    nodes = docker_brdnodes(args.project_name, args.limit_containers)
    if args.cleanup:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(cleanup_node, args.pin_root, args.project_name, node, args.command_timeout) for node in nodes]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        run(["rm", "-rf", str(args.pin_root / safe_name(args.project_name))], check=False)
        summary = {"success": True, "cleanup": True, "project_name": args.project_name, "nodes": rows}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if not nodes:
        raise SystemExit(f"ERROR: no running brdnode containers found for project {args.project_name}")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(install_one, node, args, run_dir) for node in nodes]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            status = "ok" if row.get("success") else "FAIL"
            print(
                f"[{idx}/{len(nodes)}] {status} {row.get('name')} "
                f"routes={row.get('routes_installed')} duration={row.get('duration_seconds'):.3f}s",
                flush=True,
            )

    summary = aggregate(rows, started, wall_started_at, args, run_dir)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("success", "nodes_total", "nodes_success", "nodes_failed", "routes_installed", "duration_seconds", "run_dir")}, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
