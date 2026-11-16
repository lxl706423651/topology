#!/usr/bin/env python3
#
# Purpose: test Docker-level reachability among SeedEMU brdnode containers.
# Inputs: output/docker-compose.yml, optional assignment.pkl and topology report.
# Outputs: a connectivity report JSON and a non-zero exit code on ping failure.
# Side effects: runs docker exec ping commands inside brdnode containers only.
# Expected context: run from fullMeshExample after dockerUpFullMesh.sh.

import argparse
import concurrent.futures
import json
import pickle
import re
import subprocess
import time
from pathlib import Path

import yaml


def parseArgs():
    """Parse CLI arguments for brdnode-only ping checks."""
    parser = argparse.ArgumentParser(description="Ping-test brdnode containers from SeedEMU compose output.")
    parser.add_argument("--compose", default="output/docker-compose.yml", help="Docker compose file.")
    parser.add_argument("--assignment", default="assignment.pkl", help="assignment.pkl path for root ASN mapping.")
    parser.add_argument(
        "--report",
        default="real_topology_1078_connected.json.report.json",
        help="Topology completion report with changes.root_asn.",
    )
    parser.add_argument("--output", default="connectivity_report.json", help="Connectivity report JSON path.")
    parser.add_argument("--mode", choices=["root", "all-pairs"], default="root", help="Ping matrix mode.")
    parser.add_argument("--root-service", default=None, help="Explicit root brdnode service name.")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent ping workers.")
    parser.add_argument("--timeout", type=int, default=2, help="Per-ping timeout seconds passed to ping -W.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per ping pair.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Delay between retries in seconds.")
    parser.add_argument("--wait", type=float, default=0.0, help="Initial wait before testing, useful for BGP convergence.")
    parser.add_argument("--limit", type=int, default=None, help="Limit brdnode targets for smoke tests.")
    parser.add_argument(
        "--target-address",
        choices=["dummy", "first-ip"],
        default="dummy",
        help="Address to ping for each brdnode. dummy uses the router dummy0 /32; first-ip uses the first compose IP.",
    )
    return parser.parse_args()


def loadCompose(path):
    """Load Docker compose YAML."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not data or "services" not in data:
        raise ValueError(f"{path} does not contain compose services")
    return data


def firstIpv4Address(service):
    """Return the first static ipv4_address for a compose service."""
    networks = service.get("networks") or {}
    for _network_name, config in networks.items():
        if isinstance(config, dict) and config.get("ipv4_address"):
            return config["ipv4_address"]
    return None


def serviceBuildContext(service):
    """Return the generated node build context from an original SeedEMU service."""
    build = service.get("build")
    if isinstance(build, str):
        return build
    if isinstance(build, dict):
        return build.get("context")
    return None


def resolveRelativePath(base_dir, path_text):
    """Resolve a compose-relative path without requiring it to exist first."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def nodeDirectoryForService(compose_path, service_name, service):
    """Find the generated node directory from build context, fast-compose volumes, or service name."""
    compose_dir = Path(compose_path).resolve().parent
    context = serviceBuildContext(service)
    if context:
        return resolveRelativePath(compose_dir, context)

    for volume in service.get("volumes") or []:
        if not isinstance(volume, str) or ":" not in volume:
            continue
        source, target = volume.split(":", 1)
        if target.split(":", 1)[0] == "/start.sh":
            return resolveRelativePath(compose_dir, source).parent

    return compose_dir / service_name


def dummyIpv4Address(node_dir):
    """Read the router dummy0 /32 address from the generated start script."""
    if not node_dir.exists():
        return None
    pattern = re.compile(r"ip addr add ([0-9.]+)/32 dev dummy0")
    for path in node_dir.iterdir():
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = pattern.search(content)
        if match:
            return match.group(1)
    return None


def selectTargetAddress(compose_path, service_name, service, target_address):
    """Choose the address used by ping tests for this brdnode service."""
    if target_address == "first-ip":
        return firstIpv4Address(service)
    return dummyIpv4Address(nodeDirectoryForService(compose_path, service_name, service))


def loadBrdNodes(compose_path, limit=None, target_address="dummy"):
    """Extract brdnode service/container/IP records and ignore rs_ix services."""
    compose = loadCompose(compose_path)
    brd_nodes = []
    for service_name, service in compose["services"].items():
        if not service_name.startswith("brdnode_"):
            continue
        ip = selectTargetAddress(compose_path, service_name, service, target_address)
        if not ip:
            continue
        labels = service.get("labels") or {}
        brd_nodes.append(
            {
                "service": service_name,
                "container": service.get("container_name", service_name),
                "ip": ip,
                "asn": str(labels.get("org.seedsecuritylabs.seedemu.meta.asn", "")),
                "node": str(labels.get("org.seedsecuritylabs.seedemu.meta.nodename", "")),
            }
        )
    brd_nodes.sort(key=lambda item: item["service"])
    if limit is not None:
        brd_nodes = brd_nodes[:limit]
    return brd_nodes


def loadProviderRootCompiledAsn(report_path, assignment_path):
    """Map the original provider-star root ASN from report JSON to the compiled ASN."""
    report = Path(report_path)
    assignment = Path(assignment_path)
    if not report.exists() or not assignment.exists():
        return None
    report_data = json.loads(report.read_text(encoding="utf-8"))
    root_asn = str(report_data.get("changes", {}).get("root_asn", ""))
    if not root_asn:
        return None
    with open(assignment, "rb") as handle:
        assignment_data = pickle.load(handle)
    if root_asn not in assignment_data:
        return None
    return str(assignment_data[root_asn]["asn"])


def chooseRootNode(brd_nodes, root_service=None, root_compiled_asn=None):
    """Choose the root source node for bidirectional root-mode tests."""
    if root_service:
        for node in brd_nodes:
            if node["service"] == root_service:
                return node
        raise ValueError(f"root service {root_service} was not found in brdnode services")
    if root_compiled_asn:
        for node in brd_nodes:
            if node["asn"] == root_compiled_asn:
                return node
    if not brd_nodes:
        raise ValueError("no brdnode services found")
    return brd_nodes[0]


def buildPingPairs(brd_nodes, mode, root_node):
    """Build ping source/target pairs for root or all-pairs mode."""
    pairs = []
    if mode == "root":
        for target in brd_nodes:
            if target["service"] != root_node["service"]:
                pairs.append((root_node, target))
                pairs.append((target, root_node))
        return pairs

    for source in brd_nodes:
        for target in brd_nodes:
            if source["service"] != target["service"]:
                pairs.append((source, target))
    return pairs


def loadRunningContainers():
    """Return running Docker container names from one daemon query."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker ps failed: {result.stderr.strip()}")
    return set(result.stdout.splitlines())


def pingOnce(source, target, timeout):
    """Run one docker exec ping from source container to target IP."""
    start = time.time()
    result = subprocess.run(
        ["docker", "exec", source["container"], "ping", "-c", "1", "-W", str(timeout), target["ip"]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout + 3,
    )
    elapsed = time.time() - start
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, elapsed, output[-500:]


def pingWithRetries(source, target, timeout, retries, retry_delay):
    """Ping with retry to absorb BGP startup convergence."""
    last_elapsed = 0.0
    last_output = ""
    for attempt in range(1, retries + 1):
        try:
            ok, elapsed, output = pingOnce(source, target, timeout)
        except subprocess.TimeoutExpired as exc:
            ok = False
            elapsed = timeout + 3
            output = str(exc)
        last_elapsed = elapsed
        last_output = output
        if ok:
            return {
                "ok": True,
                "attempt": attempt,
                "elapsed": elapsed,
                "source": source["service"],
                "source_container": source["container"],
                "target": target["service"],
                "target_container": target["container"],
                "target_ip": target["ip"],
            }
        if attempt < retries:
            time.sleep(retry_delay)
    return {
        "ok": False,
        "attempt": retries,
        "elapsed": last_elapsed,
        "source": source["service"],
        "source_container": source["container"],
        "target": target["service"],
        "target_container": target["container"],
        "target_ip": target["ip"],
        "error": last_output,
    }


def main():
    """Run the selected ping matrix and write a machine-readable report."""
    args = parseArgs()
    if args.wait > 0:
        print(f"waiting {args.wait}s before ping tests")
        time.sleep(args.wait)

    brd_nodes = loadBrdNodes(args.compose, args.limit, args.target_address)
    root_compiled_asn = loadProviderRootCompiledAsn(args.report, args.assignment)
    root_node = chooseRootNode(brd_nodes, args.root_service, root_compiled_asn)

    running_containers = loadRunningContainers()
    not_running = [node for node in brd_nodes if node["container"] not in running_containers]
    if not_running:
        report = {
            "ok": False,
            "reason": "containers_not_running",
            "not_running": not_running[:50],
            "not_running_count": len(not_running),
            "brdnode_count": len(brd_nodes),
        }
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"containers not running: {len(not_running)}; wrote {args.output}")
        raise SystemExit(2)

    pairs = buildPingPairs(brd_nodes, args.mode, root_node)
    print(
        f"testing {len(pairs)} ping pairs across {len(brd_nodes)} brdnodes; "
        f"mode={args.mode}; target_address={args.target_address}; root={root_node['service']}"
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pair = {
            executor.submit(
                pingWithRetries,
                source,
                target,
                args.timeout,
                args.retries,
                args.retry_delay,
            ): (source, target)
            for source, target in pairs
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_pair), start=1):
            result = future.result()
            results.append(result)
            if index % 100 == 0 or index == len(pairs):
                failures = sum(1 for item in results if not item["ok"])
                print(f"progress {index}/{len(pairs)}; failures={failures}")

    failures = [result for result in results if not result["ok"]]
    report = {
        "ok": not failures,
        "mode": args.mode,
        "target_address": args.target_address,
        "brdnode_count": len(brd_nodes),
        "pair_count": len(pairs),
        "root": root_node,
        "failure_count": len(failures),
        "failures": failures[:200],
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"result: {'ok' if not failures else 'failed'}; failures={len(failures)}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
