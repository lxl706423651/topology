#!/usr/bin/env python3
#
# Purpose: patch generated SeedEMU brdnode BIRD configs for large RR topologies.
# Inputs: output/docker-compose.yml and generated brdnode directories.
# Outputs: updated per-node BIRD configs under the compile output directory.
# Side effects: edits generated output files only; does not touch running containers.
# Expected context: run after RR_1897.py compile and before build/up.

import argparse
import re
from collections import defaultdict
from pathlib import Path

import yaml


BIRD_CONF_FILE = "2b0ae038330eccd43095538618caee7d"
START_FILE = "082b96ec819c95ae773daebde675ef80"


def parseArgs():
    """Parse CLI arguments for generated BIRD config patching."""
    parser = argparse.ArgumentParser(description="Patch generated brdnode BIRD configs.")
    parser.add_argument("--compose", default="output/docker-compose.yml", help="SeedEMU compose file.")
    return parser.parse_args()


def readCompose(compose_path):
    """Load the Docker compose YAML emitted by SeedEMU."""
    with open(compose_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def readDummyAddress(output_dir, service_name):
    """Read a generated router dummy0 /32 from its start script."""
    start_path = output_dir / service_name / START_FILE
    if not start_path.exists():
        return None
    match = re.search(r"ip addr add ([0-9.]+)/32 dev dummy0", start_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def brdServices(compose):
    """Yield brdnode service names and definitions."""
    for service_name, service in compose.get("services", {}).items():
        if service_name.startswith("brdnode_"):
            yield service_name, service


def buildNeighborRoutes(compose_path, compose):
    """Build per-service static routes to adjacent brdnode dummy0 addresses."""
    output_dir = Path(compose_path).resolve().parent
    dummy_by_service = {}
    members_by_network = defaultdict(list)

    for service_name, service in brdServices(compose):
        dummy = readDummyAddress(output_dir, service_name)
        if dummy:
            dummy_by_service[service_name] = dummy
        for network_name, config in (service.get("networks") or {}).items():
            if "_net_" not in network_name:
                continue
            if not isinstance(config, dict) or not config.get("ipv4_address"):
                continue
            members_by_network[network_name].append((service_name, config["ipv4_address"]))

    routes_by_service = defaultdict(list)
    for members in members_by_network.values():
        if len(members) != 2:
            continue
        (left_service, left_ip), (right_service, right_ip) = members
        left_dummy = dummy_by_service.get(left_service)
        right_dummy = dummy_by_service.get(right_service)
        if left_dummy and right_dummy:
            routes_by_service[left_service].append((right_dummy, right_ip))
            routes_by_service[right_service].append((left_dummy, left_ip))

    return routes_by_service


def ensureDummyExport(text):
    """Ensure dummy0 is imported into t_direct and exported to BGP as local."""
    if "protocol direct dummy_nets" in text:
        return text
    return (
        text.rstrip()
        + "\n\nprotocol direct dummy_nets {\n"
        + "    ipv4 { table t_direct; import all; };\n"
        + "    interface \"dummy0\";\n"
        + "}\n"
    )


def tuneOspf(text):
    """Use stable point-to-point OSPF timers for internal net_* links."""
    return re.sub(
        r'interface "(net_[^"]+)" \{[^}]+\};',
        r'interface "\1" { hello 10; dead 40; type pointopoint; retransmit 20; };',
        text,
    )


def addIbgpNextHopSelf(text):
    """Add next-hop-self to RR-to-client and RR-mesh iBGP sessions."""
    lines = text.splitlines()
    output = []
    in_cli = False
    in_ipv4 = False
    cli_has_next = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("protocol bgp Ibgp_to_cli_") or stripped.startswith("protocol bgp Ibgp_mesh_"):
            in_cli = True
            in_ipv4 = False
            cli_has_next = False
        elif in_cli and stripped.startswith("protocol "):
            in_cli = False
            in_ipv4 = False
            cli_has_next = False

        if in_cli and stripped == "ipv4 {":
            in_ipv4 = True
        if in_cli and in_ipv4 and stripped == "next hop self;":
            cli_has_next = True

        output.append(line)
        if in_cli and in_ipv4 and stripped == "igp table t_ospf;" and not cli_has_next:
            output.append("        next hop self;")
            cli_has_next = True

        if in_cli and in_ipv4 and stripped == "};":
            in_ipv4 = False

    return "\n".join(output) + "\n"


def removeStaticNeighborBlock(text):
    """Remove an existing static_ibgp_neighbors protocol block."""
    lines = text.splitlines()
    output = []
    skipping = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("protocol static static_ibgp_neighbors"):
            skipping = True
            depth = line.count("{") - line.count("}")
            continue
        if skipping:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
            continue
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def removeNamedProtocolBlock(text, protocol_prefix):
    """Remove a generated named protocol block by its first-line prefix."""
    lines = text.splitlines()
    output = []
    skipping = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(protocol_prefix):
            skipping = True
            depth = line.count("{") - line.count("}")
            continue
        if skipping:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
            continue
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def addStaticNeighborRoutes(text, routes):
    """Append BIRD static routes for adjacent iBGP loopback resolution."""
    text = removeStaticNeighborBlock(text)
    if not routes:
        return text

    lines = [
        "",
        "protocol static static_ibgp_neighbors {",
        "    ipv4 { table t_ospf; };",
    ]
    for dummy_ip, via_ip in sorted(set(routes)):
        lines.append(f"    route {dummy_ip}/32 via {via_ip};")
    lines.append("}")
    return text.rstrip() + "\n" + "\n".join(lines) + "\n"


def addOspfDummyToBgpPipe(text):
    """Export AS-local OSPF dummy0 routes into BGP as local routes."""
    if "define LOCAL_COMM" not in text:
        return removeNamedProtocolBlock(text, "protocol pipe pipe_ospf_dummy_to_bgp")
    text = removeNamedProtocolBlock(text, "protocol pipe pipe_ospf_dummy_to_bgp")
    block = """
protocol pipe pipe_ospf_dummy_to_bgp {
    table t_ospf;
    peer table t_bgp;
    import none;
    export filter {
        if net ~ [ 10.0.0.0/8{8,32} ] then {
            bgp_large_community.add(LOCAL_COMM);
            bgp_local_pref = 40;
            accept;
        }
        reject;
    };
}
"""
    return text.rstrip() + "\n" + block


def patchBirdConfig(path, routes):
    """Patch one generated BIRD config and return whether it changed."""
    original = path.read_text(encoding="utf-8")
    patched = ensureDummyExport(original)
    patched = tuneOspf(patched)
    patched = addIbgpNextHopSelf(patched)
    patched = addStaticNeighborRoutes(patched, routes)
    patched = addOspfDummyToBgpPipe(patched)
    if patched == original:
        return False
    path.write_text(patched, encoding="utf-8")
    return True


def main():
    """Patch every generated brdnode BIRD config."""
    args = parseArgs()
    compose_path = Path(args.compose)
    compose = readCompose(compose_path)
    output_dir = compose_path.resolve().parent
    routes_by_service = buildNeighborRoutes(compose_path, compose)

    patched = 0
    brd_count = 0
    route_count = 0
    for service_name, _service in brdServices(compose):
        brd_count += 1
        routes = routes_by_service.get(service_name, [])
        route_count += len(routes)
        bird_path = output_dir / service_name / BIRD_CONF_FILE
        if bird_path.exists() and patchBirdConfig(bird_path, routes):
            patched += 1

    print(f"patched_brd_configs {patched}/{brd_count}; static_neighbor_routes {route_count}")


if __name__ == "__main__":
    main()
