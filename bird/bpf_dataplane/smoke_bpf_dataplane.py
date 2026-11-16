#!/usr/bin/env python3
"""Self-contained TC/BPF dataplane feasibility smoke test.

Topology:

    h1 -- r1 -- r2 -- h2

Routers keep Linux ip_forward disabled and do not have remote Linux FIB
routes. TC BPF programs on r1/r2 perform LPM lookup from BPF maps and redirect
packets with bpf_redirect_neigh().
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR / "tc_lpm_router.bpf.c"
BUILD_DIR = SCRIPT_DIR / "build"
OBJ = BUILD_DIR / "tc_lpm_router.bpf.o"
PIN_ROOT = Path("/sys/fs/bpf/seedemu_bpf_smoke")
NAMESPACES = ("h1", "r1", "r2", "h2")
BPFT = "bpftool"


def run(cmd: list[str] | str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    text_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"$ {text_cmd}", flush=True)
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def need_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("ERROR: run as root, e.g. sudo python3 smoke_bpf_dataplane.py")


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
        proc = subprocess.run([candidate, "version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            return candidate
    raise SystemExit("ERROR: no usable bpftool found; Ubuntu's /usr/sbin/bpftool wrapper may not support this custom kernel")


def clang_bin() -> str:
    for name in ("clang", "clang-18", "clang-17", "clang-16"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("ERROR: clang not found; install clang/llvm first")


def compile_bpf(force: bool = False) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if OBJ.exists() and not force and OBJ.stat().st_mtime >= SRC.stat().st_mtime:
        print(f"Using existing object: {OBJ}")
        return
    clang = clang_bin()
    include_args = ["-I/usr/include", "-I/usr/include/x86_64-linux-gnu"]
    run([clang, "-O2", "-g", "-target", "bpf", "-D__TARGET_ARCH_x86", "-Wall", "-Werror", *include_args, "-c", str(SRC), "-o", str(OBJ)])


def cleanup() -> None:
    for ns in NAMESPACES:
        run(["ip", "netns", "del", ns], check=False)
    run(["rm", "-rf", str(PIN_ROOT)], check=False)


def setup_netns() -> None:
    cleanup()
    for ns in NAMESPACES:
        run(["ip", "netns", "add", ns])
        run(["ip", "-n", ns, "link", "set", "lo", "up"])

    links = [
        ("h1eth", "r1h1", "h1", "r1"),
        ("r1r2", "r2r1", "r1", "r2"),
        ("r2h2", "h2eth", "r2", "h2"),
    ]
    for a, b, nsa, nsb in links:
        run(["ip", "link", "add", a, "type", "veth", "peer", "name", b])
        run(["ip", "link", "set", a, "netns", nsa])
        run(["ip", "link", "set", b, "netns", nsb])
        run(["ip", "-n", nsa, "link", "set", a, "up"])
        run(["ip", "-n", nsb, "link", "set", b, "up"])

    addrs = [
        ("h1", "h1eth", "10.0.1.2/24"),
        ("r1", "r1h1", "10.0.1.1/24"),
        ("r1", "r1r2", "10.0.12.1/24"),
        ("r2", "r2r1", "10.0.12.2/24"),
        ("r2", "r2h2", "10.0.3.1/24"),
        ("h2", "h2eth", "10.0.3.2/24"),
    ]
    for ns, dev, addr in addrs:
        run(["ip", "-n", ns, "addr", "add", addr, "dev", dev])

    run(["ip", "-n", "h1", "route", "add", "default", "via", "10.0.1.1"])
    run(["ip", "-n", "h2", "route", "add", "default", "via", "10.0.3.1"])
    for ns in ("r1", "r2"):
        run(["ip", "netns", "exec", ns, "sysctl", "-q", "-w", "net.ipv4.ip_forward=0"])


def iface_index(ns: str, dev: str) -> int:
    out = run(["ip", "-n", ns, "-j", "link", "show", "dev", dev], capture=True).stdout
    return int(json.loads(out)[0]["ifindex"])


def iface_mac(ns: str, dev: str) -> str:
    out = run(["ip", "-n", ns, "-j", "link", "show", "dev", dev], capture=True).stdout
    return json.loads(out)[0]["address"]


def pin_dir(router: str) -> Path:
    return PIN_ROOT / router


def prog_path(router: str) -> Path:
    return pin_dir(router) / "tc_router"


def map_path(router: str, map_name: str) -> Path:
    return pin_dir(router) / "maps" / map_name


def load_router(router: str) -> None:
    pdir = pin_dir(router)
    run(["rm", "-rf", str(pdir)], check=False)
    (pdir / "maps").mkdir(parents=True, exist_ok=True)
    run([BPFT, "prog", "loadall", str(OBJ), str(pdir), "pinmaps", str(pdir / "maps")])
    if not prog_path(router).exists():
        pins = sorted(str(p) for p in pdir.rglob("*"))
        raise RuntimeError(f"tc_router program not pinned under {pdir}; pins={pins}")


def attach(router: str, dev: str) -> None:
    netns_path = f"/run/netns/{router}"
    run(["nsenter", f"--net={netns_path}", "--", "tc", "qdisc", "replace", "dev", dev, "clsact"])
    run([
        "nsenter",
        f"--net={netns_path}",
        "--",
        "tc",
        "filter",
        "replace",
        "dev",
        dev,
        "ingress",
        "pref",
        "10",
        "bpf",
        "da",
        "pinned",
        str(prog_path(router)),
    ])


def hex_bytes(buf: bytes) -> list[str]:
    return [f"{b:02x}" for b in buf]


def update_route(router: str, prefix: str, ifindex: int, nexthop: str) -> None:
    net = ipaddress.ip_network(prefix, strict=False)
    key = struct.pack("<I", net.prefixlen) + socket.inet_aton(str(net.network_address))
    value = struct.pack("<I", ifindex) + socket.inet_aton(nexthop)
    run([
        BPFT,
        "map",
        "update",
        "pinned",
        str(map_path(router, "routes")),
        "key",
        "hex",
        *hex_bytes(key),
        "value",
        "hex",
        *hex_bytes(value),
    ])


def add_static_neigh(ns: str, ip: str, peer_ns: str, peer_dev: str, dev: str) -> None:
    mac = iface_mac(peer_ns, peer_dev)
    run(["ip", "-n", ns, "neigh", "replace", ip, "lladdr", mac, "dev", dev, "nud", "permanent"])


def setup_bpf() -> None:
    for router in ("r1", "r2"):
        load_router(router)

    r1_r1r2 = iface_index("r1", "r1r2")
    r1_r1h1 = iface_index("r1", "r1h1")
    r2_r2r1 = iface_index("r2", "r2r1")
    r2_r2h2 = iface_index("r2", "r2h2")

    for dev in ("r1h1", "r1r2"):
        attach("r1", dev)
    for dev in ("r2r1", "r2h2"):
        attach("r2", dev)

    update_route("r1", "10.0.3.0/24", r1_r1r2, "10.0.12.2")
    update_route("r1", "10.0.1.0/24", r1_r1h1, "10.0.1.2")
    update_route("r2", "10.0.1.0/24", r2_r2r1, "10.0.12.1")
    update_route("r2", "10.0.3.0/24", r2_r2h2, "10.0.3.2")

    add_static_neigh("r1", "10.0.12.2", "r2", "r2r1", "r1r2")
    add_static_neigh("r1", "10.0.1.2", "h1", "h1eth", "r1h1")
    add_static_neigh("r2", "10.0.12.1", "r1", "r1r2", "r2r1")
    add_static_neigh("r2", "10.0.3.2", "h2", "h2eth", "r2h2")


def show_state() -> None:
    for ns in ("r1", "r2"):
        print(f"\n## {ns} Linux routes")
        run(["ip", "-n", ns, "route", "show"], check=False)
        print(f"\n## {ns} remote route get checks")
        remote = "10.0.3.2" if ns == "r1" else "10.0.1.2"
        run(["ip", "-n", ns, "route", "get", remote], check=False)
        print(f"\n## {ns} BPF routes")
        run([BPFT, "map", "dump", "pinned", str(map_path(ns, "routes"))], check=False)


def run_ping() -> None:
    # First packet may still be consumed by neighbor resolution on some kernels,
    # despite static entries. Run a short warmup before the measured check.
    run(["ip", "netns", "exec", "h1", "ping", "-c", "1", "-W", "1", "10.0.3.2"], check=False)
    res = run(["ip", "netns", "exec", "h1", "ping", "-c", "3", "-W", "1", "10.0.3.2"], check=False)
    if res.returncode != 0:
        raise RuntimeError("BPF dataplane ping failed")
    run(["ip", "netns", "exec", "h2", "ping", "-c", "3", "-W", "1", "10.0.1.2"])


def main() -> int:
    global BPFT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", action="store_true", help="Only remove smoke namespaces and pinned BPF objects.")
    parser.add_argument("--keep", action="store_true", help="Keep namespaces and pinned BPF objects after success.")
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()

    need_root()
    for name in ("ip", "tc", "ping", "rm"):
        tool(name)
    BPFT = usable_bpftool()
    print(f"Using bpftool: {BPFT}")

    if args.cleanup:
        cleanup()
        return 0

    try:
        compile_bpf(force=args.force_compile)
        setup_netns()
        setup_bpf()
        show_state()
        started = time.monotonic()
        run_ping()
        elapsed = time.monotonic() - started
        print(f"\nSUCCESS: TC/BPF dataplane forwarded ping without remote Linux FIB routes in {elapsed:.3f}s")
        return 0
    finally:
        if not args.keep:
            cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
