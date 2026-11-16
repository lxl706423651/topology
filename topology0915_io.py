"""Minimal literal-only AS/IX topology format for cutGragh/createNetwork0915."""
from __future__ import annotations

import ast
import math
import re
from collections import defaultdict
from pathlib import Path

FIELDS = ("as_ix", "as_as_ix_edges", "rs_groups")


def positive_integer(value, label):
    if type(value) is not int or not 0 < value <= 4294967295:
        raise ValueError(f"{label} must be a positive 32-bit integer: {value!r}")
    return value


def ix_key(ix):
    if not isinstance(ix, str) or not re.fullmatch(r"(?:pdb|caida-ixps(?:-\d+)?):[1-9]\d*", ix):
        raise ValueError(f"Invalid canonical IX key: {ix!r}")
    return ix


def validate(data):
    if set(data) != set(FIELDS):
        raise ValueError(f"Expected exactly these fields: {FIELDS}")
    as_ix = data["as_ix"]
    if not isinstance(as_ix, dict) or not as_ix:
        raise ValueError("as_ix must contain the original AS-IX router inventory")
    for asn, ixes in as_ix.items():
        positive_integer(asn, "ASN")
        if not isinstance(ixes, list) or not ixes or len(set(ixes)) != len(ixes):
            raise ValueError(f"AS{asn}: expected a nonempty unique IX list")
        for ix in ixes:
            ix_key(ix)
    relations, seen = {}, set()
    for row in data["as_as_ix_edges"]:
        if len(row) != 4:
            raise ValueError(f"Invalid quadruple: {row!r}")
        a, b, ix, rel = row
        positive_integer(a, "ASN")
        positive_integer(b, "ASN")
        if a == b or a not in as_ix or b not in as_ix:
            raise ValueError(f"Unknown/self AS in quadruple: {row!r}")
        if type(rel) is not int or rel not in (-1, 0):
            raise ValueError(f"Expected -1=provider->customer or 0=peer: {row!r}")
        if ix is not None:
            ix_key(ix)
            if ix not in as_ix[a] or ix not in as_ix[b]:
                raise ValueError(f"Quadruple endpoint has no original router at IX: {row!r}")
        pair = tuple(sorted((a, b)))
        relation = (a, b, -1) if rel == -1 else (*pair, 0)
        if pair in relations and relations[pair] != relation:
            raise ValueError(f"Conflicting relationship for {pair}")
        relations[pair] = relation
        record = (*pair, ix, rel)
        if record in seen:
            raise ValueError(f"Duplicate quadruple: {row!r}")
        seen.add(record)
    seen_groups = set()
    for row in data["rs_groups"]:
        if len(row) != 3:
            raise ValueError(f"Invalid RS triple: {row!r}")
        members, ix, rs = row
        ix_key(ix)
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError("RS group must contain at least two distinct AS members")
        for asn in members:
            positive_integer(asn, "RS member ASN")
            if asn not in as_ix or ix not in as_ix[asn]:
                raise ValueError(f"RS member AS{asn} has no router at {ix}")
        if rs is not None:
            positive_integer(rs, "RS ASN")
            if rs not in as_ix:
                raise ValueError(f"External AS{rs} in RS triple")
        key = (tuple(sorted(members)), ix, rs)
        if key in seen_groups:
            raise ValueError(f"Duplicate RS triple: {key}")
        seen_groups.add(key)
    return data


def statistics(data):
    return {"AS": len(data["as_ix"]),
            "router": sum(len(ixes) for ixes in data["as_ix"].values()),
            "IX": len({ix for ixes in data["as_ix"].values() for ix in ixes}),
            "peering": len(data["as_as_ix_edges"]) + sum(math.comb(len(members), 2) for members, _, _ in data["rs_groups"])}


def format_topology(data):
    validate(data)
    counts = statistics(data)
    header = "# " + ", ".join(f"{name}={count}" for name, count in counts.items())
    return header + "\n" + "\n".join(f"{key}: {data[key]!r}" for key in FIELDS) + "\n"


def read_topology(path):
    """Read key: Python-literal lines safely, never executable Python code."""
    data, header = {}, None
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.fullmatch(r"# AS=(\d+), router=(\d+), IX=(\d+), peering=(\d+)", line)
            if match:
                if header is not None:
                    raise ValueError("Duplicate statistics header")
                header = dict(zip(("AS", "router", "IX", "peering"), map(int, match.groups())))
            continue
        if ": " not in line:
            raise ValueError(f"{path}:{number}: expected key: value")
        key, value = line.split(": ", 1)
        if key in data:
            raise ValueError(f"{path}:{number}: duplicate key {key}")
        data[key] = ast.literal_eval(value)
    validate(data)
    if header is None or header != statistics(data):
        raise ValueError("Missing/incorrect AS/router/IX/peering statistics header")
    return data


def compact_selection(payload):
    """Keep full router inventory and relationships, drop audit metadata.

    Original RS support pairs without a retained triple are represented as
    peer quadruples for simulation. No new pair is inferred in this conversion.
    """
    if payload["status"] != "EXACT":
        raise ValueError("Cannot export a non-exact result")
    edges = {tuple(row) for row in payload["p2c"] + payload["p2p"]}
    triples = [(list(members), ix, rs) for members, ix, rs in payload["rs_peer"]]
    groups_by_member = defaultdict(list)
    for members, ix, _ in triples:
        group = set(members)
        for member in members:
            groups_by_member[ix, member].append(group)
    direct = {(min(a, b), max(a, b), ix) for a, b, ix, _ in edges}
    for a, b, ix in payload["rs_pair_edges"]:
        if (min(a, b), max(a, b), ix) in direct:
            continue
        if not any(b in group for group in groups_by_member[ix, a]):
            edges.add((a, b, ix, 0))
    data = {"as_ix": {int(a): list(ixes) for a, ixes in payload["as_router_ix"].items()},
            "as_as_ix_edges": sorted(edges, key=lambda row: (row[0], row[1], row[2] or "", row[3])),
            "rs_groups": triples}
    validate(data)
    if statistics(data)["router"] != payload["requested_router_count"]:
        raise ValueError("Compact output lost original AS routers")
    return data
