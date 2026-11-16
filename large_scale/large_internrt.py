#!/usr/bin/env python3
"""Compile a selected RR Internet topology with the seedpy310 environment.

Inputs: topology/input/assignment.pkl and real_topology_<scale>.txt. Outputs:
output/docker-compose.yml, node build contexts and topology-summary.json.
Only local compiler files are created; Docker and BIRD are started separately.
Adapted from B55_anypro/largescale/topology/RR_214_example.py, without AnyPro
origins, probe hosts, fixed PoPs or a dependency on that Kubernetes example.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
from ipaddress import IPv4Network
import json
from pathlib import Path
import pickle
import pickletools
import sys

import yaml

from runExperiment import SCRIPT_DIR, loadSettings, writeJson


def readInputs(inputDirectory: Path, scale: int) -> tuple[dict, dict, dict]:
    """Read plain assignment data and the selected scale's literal topology fields."""
    files = [inputDirectory / "assignment.pkl", inputDirectory / f"real_topology_{scale}.txt"]
    raw = files[0].read_bytes()
    executable = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "OBJ", "INST", "NEWOBJ", "NEWOBJ_EX",
                  "BUILD", "PERSID", "BINPERSID"}
    if any(op.name in executable for op, _, _ in pickletools.genops(raw)):
        raise ValueError("assignment.pkl must contain plain mapping data")
    assignment = pickle.loads(raw)
    if not isinstance(assignment, dict):
        raise ValueError("assignment.pkl must be a mapping")
    topology = {}
    for number, line in enumerate(files[1].read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or key.strip() in topology:
            raise ValueError(f"invalid or repeated topology field on line {number}")
        topology[key.strip()] = ast.literal_eval(value.strip())
    required = {"ixps", "transit_asns", "stub_asns", "ix_ix_transit_edges", "as_as_ix_edges"}
    if set(topology) != required or topology["stub_asns"]:
        raise ValueError("expected the transit-only RR schema (ixps/transit_asns/stub_asns/edges)")
    active = set(topology["ixps"]) | set(topology["transit_asns"])
    if not active.issubset(assignment):
        raise ValueError("assignment.pkl is missing active AS/IX records")
    provenance = {path.name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in files}
    return assignment, topology, provenance


def createConnectedPairs(nodes: list[int]) -> list[tuple[int, int]]:
    """Preserve the source RR ring rule for sorted IX router IDs in nodes."""
    nodes = sorted(nodes)
    if len(nodes) < 2:
        return []
    pairs = {tuple(sorted((nodes[i], nodes[(i + 1) % len(nodes)]))) for i in range(len(nodes))}
    if len(pairs) >= 254:
        for i, node in enumerate(nodes):
            other = int(i + 1 + len(nodes) / 2) % len(nodes)
            if i != other:
                pairs.add(tuple(sorted((node, nodes[other]))))
    if len(pairs) > 254:
        raise ValueError("RR /16-to-/24 allocation supports at most 254 internal links per AS")
    return sorted(pairs)


def createTopologyPlan(inputDirectory: Path, scale: int) -> dict:
    """Validate scale, AS/IX membership, address capacity and peering directions."""
    assignment, topology, provenance = readInputs(inputDirectory, scale)
    transits, exchanges = set(topology["transit_asns"]), set(topology["ixps"])
    members = defaultdict(set)
    for left, right, asn, _ in topology["ix_ix_transit_edges"]:
        if asn not in transits or left not in exchanges or right not in exchanges:
            raise ValueError("internal edge references an unknown AS/IX")
        members[asn].update((left, right))
    if set(members) != transits:
        raise ValueError("each transit AS must have an explicit IX router")
    active = transits | exchanges
    mapped = {key: int(assignment[key]["asn"]) for key in active}
    if len(set(mapped.values())) != len(active) or any(value <= 0 for value in mapped.values()):
        raise ValueError("active AS/IX assignment IDs must be positive and unique")
    prefixes = {key: IPv4Network(assignment[key]["ipv4"]) for key in active}
    if any(prefix.prefixlen != 16 for prefix in prefixes.values()):
        raise ValueError("assignment prefixes must be /16")
    if len(set(prefixes.values())) != len(active):
        raise ValueError("active assignment prefixes must not overlap")
    loopback = IPv4Network("10.0.0.0/16")
    if any(prefix.overlaps(loopback) for prefix in prefixes.values()):
        raise ValueError("assignment uses reserved loopback 10.0.0.0/16; regenerate it with build_assignment.py")
    for asn, ixNames in members.items():
        for ix in ixNames:
            address = mapped[asn]
            if address in {0, prefixes[ix].num_addresses - 1, mapped[ix]} or address >= prefixes[ix].num_addresses:
                raise ValueError("router IX address is outside its prefix or collides with the route server")
    for ix in exchanges:
        if not 0 < mapped[ix] < prefixes[ix].num_addresses - 1:
            raise ValueError("route-server IX address is outside its prefix")
    routerCount = sum(map(len, members.values()))
    if routerCount + len(exchanges) != scale:
        raise ValueError(f"filename scale {scale} differs from router+IX count {routerCount + len(exchanges)}")
    links = {asn: createConnectedPairs([mapped[ix] for ix in ixNames]) for asn, ixNames in members.items()}
    peerings = {}
    for left, right, ix, relationship in topology["as_as_ix_edges"]:
        relation = int(relationship)
        if left not in transits or right not in transits or ix not in exchanges or left == right or relation not in {-1, 0}:
            raise ValueError("invalid AS peering endpoints or relationship")
        if ix not in members[left] or ix not in members[right]:
            raise ValueError("BGP endpoint has no physical IX router")
        a, b = mapped[left], mapped[right]
        key = (mapped[ix], min(a, b), max(a, b))
        value = (relation, a if relation == -1 else None)
        if key in peerings and peerings[key] != value:
            raise ValueError("conflicting BGP relationships on one IX")
        peerings[key] = value
    return {"assignment": assignment, "topology": topology, "mapped": mapped, "members": dict(members),
            "loopback_range": str(loopback),
            "links": links, "peerings": peerings, "provenance": provenance,
            "counts": {"scale": scale, "autonomous_systems": len(transits), "routers": routerCount,
                       "route_servers": len(exchanges), "bird_nodes": scale,
                       "internal_links": sum(map(len, links.values())), "unique_peerings": len(peerings)}}


def buildEmulator(plan: dict):
    """Construct the base RR emulator from plan using the environment's SeedEMU package."""
    from seedemu.core import Emulator
    from seedemu.layers import Base, Ebgp, Ibgp, Ospf, PeerRelationship, Routing

    emu, base, ebgp = Emulator(), Base(), Ebgp()
    assignment, topology, mapped = plan["assignment"], plan["topology"], plan["mapped"]
    for original in sorted(topology["ixps"], key=mapped.get):
        ix, prefix = mapped[original], assignment[original]["ipv4"]
        exchange = base.createInternetExchange(ix, prefix, rsAddress=str(IPv4Network(prefix)[ix]))
        exchange.getPeeringLan().setDisplayName(f"IX-{ix} {original}")
    for original in sorted(topology["transit_asns"], key=mapped.get):
        asn = mapped[original]
        transit = base.createAutonomousSystem(asn)
        ixNames = sorted(plan["members"][original], key=mapped.get)
        routers = {}
        for ixName in ixNames:
            ix = mapped[ixName]
            routers[ix] = transit.createRouter(f"r{ix}").joinNetwork(
                f"ix{ix}", str(IPv4Network(assignment[ixName]["ipv4"])[asn]))
            routers[ix].addSoftware("iperf3")
        routers[mapped[ixNames[0]]].makeRouteReflector(True)
        subnets = list(IPv4Network(assignment[original]["ipv4"]).subnets(new_prefix=24))
        for index, (left, right) in enumerate(plan["links"][original], 1):
            name = f"net_{left}_{right}"
            transit.createNetwork(name, str(subnets[index]))
            routers[left].joinNetwork(name)
            routers[right].joinNetwork(name)
    for (ix, left, right), (relation, provider) in sorted(plan["peerings"].items()):
        a, b = (left, right) if relation == 0 or provider == left else (right, left)
        ebgp.addPrivatePeeringByRouters(ix, a, f"r{ix}", b, f"r{ix}",
                                      PeerRelationship.Provider if relation == -1 else PeerRelationship.Peer)
    for layer in (base, Routing(loopback_range=plan["loopback_range"]), ebgp, Ibgp(), Ospf()):
        emu.addLayer(layer)
    return emu


def compileTopology(plan: dict, output: Path, platform: str, project: str, overwrite: bool = False) -> None:
    """Render plan and compile to output; overwrite only a recognized generated directory."""
    from seedemu.compiler import Docker, Platform
    import seedemu

    output = output.expanduser().resolve()
    source = Path(plan["provenance"]["assignment.pkl"]["path"]).parent
    if output == SCRIPT_DIR or output in SCRIPT_DIR.parents or output == source or output in source.parents:
        raise ValueError("output must not replace the source or input directory")
    if output.is_relative_to(source):
        raise ValueError("output must not be placed inside the input directory")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite for a generated output directory")
        if not (output / "docker-compose.yml").is_file() or not (output / "topology-summary.json").is_file():
            raise ValueError("--overwrite requires a directory generated by large_internrt.py")
    output.parent.mkdir(parents=True, exist_ok=True)
    emu = buildEmulator(plan)
    emu.render()
    emu.compile(Docker(platform=Platform.AMD64 if platform == "amd" else Platform.ARM64), str(output), override=overwrite)
    compose = output / "docker-compose.yml"
    data = yaml.safe_load(compose.read_text())
    data["name"] = project
    compose.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    writeJson(output / "topology-summary.json", {"counts": plan["counts"], "inputs": plan["provenance"],
                                                "seedemu_package": seedemu.__file__, "platform": platform,
                                                "project": project, "loopback_range": plan["loopback_range"],
                                                "routing_preparation": "manual; see README.md"})


def compileSelectedTopology() -> int:
    """Parse CLI selection, validate inputs and optionally generate the Docker output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale", nargs="?", type=int, help="Select topology/input/real_topology_<scale>.txt")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--platform", choices=["amd", "arm"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = loadSettings(args.config)
    scale = args.scale if args.scale is not None else settings["topology"]["scale"]
    plan = createTopologyPlan(Path(settings["topology"]["input"]), scale)
    print(json.dumps(plan["counts"], ensure_ascii=False), flush=True)
    if not args.check_only:
        compileTopology(plan, args.output or Path(settings["topology"]["output"]),
                        args.platform or settings["topology"]["platform"], settings["docker"]["project"], args.overwrite)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(compileSelectedTopology())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
