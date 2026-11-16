#!/usr/bin/env python3
#
# Purpose: read a real_topology_*.txt/json file, add or promote AS-AS routing
# relationships without adding IX memberships, and write an updated JSON
# topology plus an audit report.
# Inputs: a SeedEMU topology file with ixps/transit_asns/stub_asns/
# ix_ix_transit_edges/as_as_ix_edges.
# Outputs: a connected topology JSON and a .report.json file.
# Side effects: writes only the requested output files.
# Expected context: run from fullMeshExample or pass absolute paths.

import argparse
import ast
import copy
import json
import re
from collections import defaultdict
from pathlib import Path


TOPOLOGY_KEYS = [
    "ixps",
    "transit_asns",
    "stub_asns",
    "ix_ix_transit_edges",
    "as_as_ix_edges",
]


def parseArgs():
    """Parse CLI arguments for the topology completion step."""
    parser = argparse.ArgumentParser(
        description="Add a low-cost provider-star backbone to a SeedEMU topology."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="real_topology_1078.txt",
        help="Input real_topology_*.txt or .json path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="real_topology_1078_connected.json",
        help="Output JSON topology path.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Report JSON path. Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--root-asn",
        default=None,
        help="Provider backbone root ASN. Defaults to the ASN with the most IX memberships.",
    )
    parser.add_argument(
        "--strategy",
        choices=["provider-star", "components"],
        default="provider-star",
        help="provider-star favors ping reachability; components adds only graph component links.",
    )
    return parser.parse_args()


def readTopology(path):
    """Read a topology from JSON or the legacy key: Python-literal text format."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if source.suffix == ".json":
        raw = json.loads(text)
    else:
        content = re.sub(r"^(\w+):", r'"\1":', text, flags=re.MULTILINE)
        lines = content.splitlines()
        if not lines:
            raise ValueError(f"{source} is empty")
        content = "{" + "\n".join([line + "," for line in lines[:-1]] + [lines[-1]]) + "}"
        raw = ast.literal_eval(content)
    return normalizeTopology(raw)


def normalizeTopology(raw):
    """Return a topology dict with stable string ASNs and list edge records."""
    topology = {}
    for key in TOPOLOGY_KEYS:
        topology[key] = copy.deepcopy(raw.get(key, []))

    topology["ixps"] = [str(ixp) for ixp in topology["ixps"]]
    topology["transit_asns"] = [str(asn) for asn in topology["transit_asns"]]
    topology["stub_asns"] = [str(asn) for asn in topology["stub_asns"]]
    topology["ix_ix_transit_edges"] = [
        [str(ix_a), str(ix_b), str(asn), int(rel)]
        for ix_a, ix_b, asn, rel in topology["ix_ix_transit_edges"]
    ]
    topology["as_as_ix_edges"] = [
        [str(as_a), str(as_b), str(ix), str(rel)]
        for as_a, as_b, ix, rel in topology["as_as_ix_edges"]
    ]
    return topology


def writeJson(path, data):
    """Write a deterministic UTF-8 JSON file."""
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def buildIxByAs(topology):
    """Build AS -> IX membership from ix_ix_transit_edges."""
    ix_by_as = defaultdict(set)
    for ix_a, ix_b, asn, _rel in topology["ix_ix_transit_edges"]:
        ix_by_as[str(asn)].add(str(ix_a))
        ix_by_as[str(asn)].add(str(ix_b))
    for asn in topology["transit_asns"]:
        ix_by_as[str(asn)]
    return ix_by_as


def buildAsGraph(topology):
    """Build an undirected AS graph from as_as_ix_edges."""
    graph = {str(asn): set() for asn in topology["transit_asns"] + topology["stub_asns"]}
    for as_a, as_b, _ix, _rel in topology["as_as_ix_edges"]:
        if as_a in graph and as_b in graph:
            graph[as_a].add(as_b)
            graph[as_b].add(as_a)
    return graph


def buildBrdGraph(topology):
    """Build an undirected graph over brdnode-level nodes represented as (asn, ix)."""
    ix_by_as = buildIxByAs(topology)
    nodes = set()
    graph = defaultdict(set)

    for asn, ixs in ix_by_as.items():
        ordered_ixs = sorted(ixs)
        for ix in ordered_ixs:
            nodes.add((asn, ix))
        if ordered_ixs:
            anchor = (asn, ordered_ixs[0])
            for ix in ordered_ixs[1:]:
                node = (asn, ix)
                graph[anchor].add(node)
                graph[node].add(anchor)

    for as_a, as_b, ix, _rel in topology["as_as_ix_edges"]:
        node_a = (as_a, ix)
        node_b = (as_b, ix)
        if node_a in nodes and node_b in nodes:
            graph[node_a].add(node_b)
            graph[node_b].add(node_a)

    for node in nodes:
        graph[node]
    return graph


def connectedComponents(graph):
    """Return graph connected components sorted by descending size."""
    seen = set()
    components = []
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def edgeIndex(topology):
    """Index existing AS-AS edges by unordered AS pair and exact AS/IX tuple."""
    by_pair = defaultdict(list)
    exact = set()
    provider_exact = set()
    for edge in topology["as_as_ix_edges"]:
        as_a, as_b, ix, rel = edge
        by_pair[frozenset([as_a, as_b])].append(edge)
        exact.add((as_a, as_b, ix, rel))
        if rel == "-1":
            provider_exact.add((as_a, as_b, ix))
    return by_pair, exact, provider_exact


def chooseRootAs(topology, ix_by_as, requested_root=None):
    """Choose the provider backbone root ASN."""
    if requested_root:
        root = str(requested_root)
        if root not in topology["transit_asns"]:
            raise ValueError(f"root ASN {root} is not in transit_asns")
        return root
    return max(topology["transit_asns"], key=lambda asn: (len(ix_by_as[asn]), asn))


def chooseExistingPeeringIx(root, target, ix_by_as, existing_edges):
    """Pick a shared IX where the root/target pair does not already have a peering."""
    used_ixs = {edge[2] for edge in existing_edges}
    for ix in sorted(ix_by_as[root] & ix_by_as[target]):
        if ix not in used_ixs:
            return ix
    return None


def chooseExistingCommonEdge(existing_edges, root, target, ix_by_as):
    """Choose an existing shared-IX edge that can be promoted to root provider."""
    common_ixs = ix_by_as[root] & ix_by_as[target]
    candidates = [edge for edge in existing_edges if edge[2] in common_ixs]
    return sorted(candidates, key=lambda edge: (edge[2], edge[0], edge[1], edge[3]))[0] if candidates else None


def promoteProviderEdge(edge, root, target):
    """Rewrite an existing AS-AS edge as root-provider to target-customer."""
    before = list(edge)
    edge[0] = root
    edge[1] = target
    edge[3] = "-1"
    return {"before": before, "after": list(edge)}


def addProviderStar(topology, root_asn=None):
    """Add a root-provider relation to every transit AS without adding brdnodes."""
    ix_by_as = buildIxByAs(topology)
    root = chooseRootAs(topology, ix_by_as, root_asn)
    by_pair, _exact, provider_exact = edgeIndex(topology)

    added_peer_edges = []
    promoted_peer_edges = []
    reused_provider_edges = []
    skipped_targets = []

    for target in topology["transit_asns"]:
        if target == root:
            continue

        if any(edge[0] == root and edge[1] == target and edge[3] == "-1" for edge in by_pair[frozenset([root, target])]):
            reused_provider_edges.append(target)
            continue

        existing_edges = by_pair[frozenset([root, target])]
        ix = chooseExistingPeeringIx(root, target, ix_by_as, existing_edges)
        if ix is None:
            edge = chooseExistingCommonEdge(existing_edges, root, target, ix_by_as)
            if edge is not None:
                promoted_peer_edges.append(promoteProviderEdge(edge, root, target))
                continue

        if ix is None:
            skipped_targets.append(target)
            continue

        provider_edge = [root, target, ix, "-1"]
        if (root, target, ix) not in provider_exact:
            topology["as_as_ix_edges"].append(provider_edge)
            added_peer_edges.append(provider_edge)
            provider_exact.add((root, target, ix))
            by_pair[frozenset([root, target])].append(provider_edge)

    return {
        "root_asn": root,
        "added_as_as_ix_edges": added_peer_edges,
        "promoted_as_as_ix_edges": promoted_peer_edges,
        "added_ix_ix_transit_edges": [],
        "reused_provider_targets": reused_provider_edges,
        "skipped_targets": skipped_targets,
    }


def addComponentLinks(topology, root_asn=None):
    """Connect AS graph components with provider edges between existing brdnodes."""
    graph = buildAsGraph(topology)
    components = connectedComponents(graph)
    ix_by_as = buildIxByAs(topology)
    root = chooseRootAs(topology, ix_by_as, root_asn)
    root_component = next((comp for comp in components if root in comp), components[0] if components else [])

    added_peer_edges = []
    promoted_peer_edges = []
    skipped_targets = []
    if not root_component:
        return {
            "root_asn": root,
            "added_as_as_ix_edges": added_peer_edges,
            "promoted_as_as_ix_edges": promoted_peer_edges,
            "added_ix_ix_transit_edges": [],
            "skipped_targets": skipped_targets,
        }

    root_rep = root
    by_pair, _exact, provider_exact = edgeIndex(topology)
    for component in components:
        if root_rep in component:
            continue
        target = max(component, key=lambda asn: len(ix_by_as[asn]))
        existing_edges = by_pair[frozenset([root_rep, target])]
        ix = chooseExistingPeeringIx(root_rep, target, ix_by_as, existing_edges)
        if ix is None:
            edge = chooseExistingCommonEdge(existing_edges, root_rep, target, ix_by_as)
            if edge is not None:
                promoted_peer_edges.append(promoteProviderEdge(edge, root_rep, target))
                continue
        if ix is None:
            skipped_targets.append(target)
            continue
        provider_edge = [root_rep, target, ix, "-1"]
        if (root_rep, target, ix) not in provider_exact:
            topology["as_as_ix_edges"].append(provider_edge)
            added_peer_edges.append(provider_edge)
            provider_exact.add((root_rep, target, ix))

    return {
        "root_asn": root,
        "added_as_as_ix_edges": added_peer_edges,
        "promoted_as_as_ix_edges": promoted_peer_edges,
        "added_ix_ix_transit_edges": [],
        "skipped_targets": skipped_targets,
    }


def summarize(topology):
    """Summarize graph and record counts for report output."""
    as_components = connectedComponents(buildAsGraph(topology))
    brd_components = connectedComponents(buildBrdGraph(topology))
    return {
        "ixps": len(topology["ixps"]),
        "transit_asns": len(topology["transit_asns"]),
        "stub_asns": len(topology["stub_asns"]),
        "ix_ix_transit_edges": len(topology["ix_ix_transit_edges"]),
        "as_as_ix_edges": len(topology["as_as_ix_edges"]),
        "as_components": len(as_components),
        "as_component_sizes": [len(component) for component in as_components],
        "brd_nodes": len(buildBrdGraph(topology)),
        "brd_components": len(brd_components),
        "brd_component_sizes": [len(component) for component in brd_components],
    }


def main():
    """Run topology completion and write outputs."""
    args = parseArgs()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(output_path.suffix + ".report.json")

    topology = readTopology(input_path)
    before = summarize(topology)

    if args.strategy == "provider-star":
        changes = addProviderStar(topology, args.root_asn)
    else:
        changes = addComponentLinks(topology, args.root_asn)

    after = summarize(topology)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "strategy": args.strategy,
        "before": before,
        "after": after,
        "changes": changes,
    }

    writeJson(output_path, topology)
    writeJson(report_path, report)

    print(f"wrote topology: {output_path}")
    print(f"wrote report: {report_path}")
    print(
        "summary: "
        f"AS components {before['as_components']} -> {after['as_components']}, "
        f"brd components {before['brd_components']} -> {after['brd_components']}, "
        f"added provider edges {len(changes.get('added_as_as_ix_edges', []))}, "
        f"promoted provider edges {len(changes.get('promoted_as_as_ix_edges', []))}, "
        f"added IX memberships {len(changes.get('added_ix_ix_transit_edges', []))}"
    )


if __name__ == "__main__":
    main()
