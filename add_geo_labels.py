#!/usr/bin/env python3
"""Generate deterministic, Internet-like geo labels for SeedEMU Compose files.

The layout is hierarchical rather than a cloud of points around one AS center:

* ASes are assigned to macro regions using Internet-density-inspired weights.
* Each AS receives one or more real metropolitan-area PoPs according to its size.
* Local Compose networks form an intra-AS graph.  Graph-distant nodes are chosen
  as PoP seeds, while graph-near nodes tend to remain in the same PoP.
* Small ASes remain metropolitan/national, medium ASes span a region, and only
  very large backbones may extend into one adjacent macro region.
* A small deterministic offset prevents nodes at the same PoP from overlapping.
* Global networks are placed near the real-city anchor closest to their member
  medoid, so IX markers remain related to the services that use them.

The source Compose file is never modified.  By default a sibling such as
``docker-compose-geo.yml`` is written; ``--output-template '{scale}geo.yaml'``
produces names such as ``1078geo.yaml`` and ``3083geo.yaml``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterable


ASN_KEY = "org.seedsecuritylabs.seedemu.meta.asn"
ROLE_KEY = "org.seedsecuritylabs.seedemu.meta.role"
GEO_PREFIX = "org.seedsecuritylabs.seedemu.meta.geo."
NETWORK_TYPE_KEY = "org.seedsecuritylabs.seedemu.meta.type"
NETWORK_SCOPE_KEY = "org.seedsecuritylabs.seedemu.meta.scope"
NETWORK_NAME_KEY = "org.seedsecuritylabs.seedemu.meta.name"
NETWORK_PREFIX_KEY = "org.seedsecuritylabs.seedemu.meta.prefix"
NETWORK_DISPLAYNAME_KEY = "org.seedsecuritylabs.seedemu.meta.displayname"

SERVICE_RE = re.compile(r"^ {4}([^ ][^:]*):(?:\s*#.*)?$")
SERVICE_NETWORK_RE = re.compile(r"^ {12}([^ ][^:]*):(?:\s*#.*)?$")
SERVICE_NETWORK_LIST_RE = re.compile(r"^ {12}-\s*([^\s#]+)")
ASN_RE = re.compile(
    rf"^\s+{re.escape(ASN_KEY)}:\s*[\"']?([^\"'\s#]+)"
)
GEO_RE = re.compile(rf"^\s+{re.escape(GEO_PREFIX)}(?:lat|lon|source|label):")

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Metro:
    """A real metropolitan-area anchor used as a synthetic PoP location."""

    name: str
    region: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class ServiceRecord:
    """The topology fields needed from one Compose service."""

    name: str
    asn: str
    networks: tuple[str, ...]


@dataclass(frozen=True)
class NodePlacement:
    """Generated service coordinates and their containing metro/region."""

    latitude: float
    longitude: float
    metro: str
    region: str


@dataclass
class ComposeTopology:
    """A compact topology extracted without reformatting the Compose YAML."""

    services: dict[str, ServiceRecord]
    network_types: dict[str, str]


@dataclass
class GeographicLayout:
    """A union layout shared by every Compose input in one invocation."""

    placements: dict[tuple[str, str], NodePlacement]
    as_regions: dict[str, tuple[str, ...]]
    as_pops: dict[str, tuple[Metro, ...]]
    adjacency: dict[str, dict[str, set[str]]]


# These anchors approximate major carrier hotels, cloud regions, and Internet
# exchange metros.  Exact street-level positions would suggest false precision,
# so city-center coordinates are intentionally used.
METROS = (
    # North America
    Metro("Vancouver", "north_america", 49.2827, -123.1207),
    Metro("Seattle", "north_america", 47.6062, -122.3321),
    Metro("San Francisco", "north_america", 37.7749, -122.4194),
    Metro("Los Angeles", "north_america", 34.0522, -118.2437),
    Metro("Phoenix", "north_america", 33.4484, -112.0740),
    Metro("Denver", "north_america", 39.7392, -104.9903),
    Metro("Dallas", "north_america", 32.7767, -96.7970),
    Metro("Houston", "north_america", 29.7604, -95.3698),
    Metro("Chicago", "north_america", 41.8781, -87.6298),
    Metro("Minneapolis", "north_america", 44.9778, -93.2650),
    Metro("Atlanta", "north_america", 33.7490, -84.3880),
    Metro("Miami", "north_america", 25.7617, -80.1918),
    Metro("Ashburn", "north_america", 39.0438, -77.4874),
    Metro("New York", "north_america", 40.7128, -74.0060),
    Metro("Boston", "north_america", 42.3601, -71.0589),
    Metro("Toronto", "north_america", 43.6532, -79.3832),
    Metro("Montreal", "north_america", 45.5017, -73.5673),
    Metro("Mexico City", "north_america", 19.4326, -99.1332),
    # Europe
    Metro("Dublin", "europe", 53.3498, -6.2603),
    Metro("London", "europe", 51.5074, -0.1278),
    Metro("Amsterdam", "europe", 52.3676, 4.9041),
    Metro("Brussels", "europe", 50.8503, 4.3517),
    Metro("Paris", "europe", 48.8566, 2.3522),
    Metro("Madrid", "europe", 40.4168, -3.7038),
    Metro("Lisbon", "europe", 38.7223, -9.1393),
    Metro("Frankfurt", "europe", 50.1109, 8.6821),
    Metro("Zurich", "europe", 47.3769, 8.5417),
    Metro("Milan", "europe", 45.4642, 9.1900),
    Metro("Vienna", "europe", 48.2082, 16.3738),
    Metro("Prague", "europe", 50.0755, 14.4378),
    Metro("Warsaw", "europe", 52.2297, 21.0122),
    Metro("Copenhagen", "europe", 55.6761, 12.5683),
    Metro("Stockholm", "europe", 59.3293, 18.0686),
    Metro("Helsinki", "europe", 60.1699, 24.9384),
    Metro("Bucharest", "europe", 44.4268, 26.1025),
    Metro("Athens", "europe", 37.9838, 23.7275),
    # East Asia
    Metro("Beijing", "east_asia", 39.9042, 116.4074),
    Metro("Tianjin", "east_asia", 39.3434, 117.3616),
    Metro("Shanghai", "east_asia", 31.2304, 121.4737),
    Metro("Hangzhou", "east_asia", 30.2741, 120.1551),
    Metro("Shenzhen", "east_asia", 22.5431, 114.0579),
    Metro("Hong Kong", "east_asia", 22.3193, 114.1694),
    Metro("Taipei", "east_asia", 25.0330, 121.5654),
    Metro("Seoul", "east_asia", 37.5665, 126.9780),
    Metro("Busan", "east_asia", 35.1796, 129.0756),
    Metro("Tokyo", "east_asia", 35.6762, 139.6503),
    Metro("Osaka", "east_asia", 34.6937, 135.5023),
    Metro("Fukuoka", "east_asia", 33.5902, 130.4017),
    # South and Southeast Asia
    Metro("Delhi", "south_southeast_asia", 28.6139, 77.2090),
    Metro("Mumbai", "south_southeast_asia", 19.0760, 72.8777),
    Metro("Chennai", "south_southeast_asia", 13.0827, 80.2707),
    Metro("Bengaluru", "south_southeast_asia", 12.9716, 77.5946),
    Metro("Hyderabad", "south_southeast_asia", 17.3850, 78.4867),
    Metro("Dhaka", "south_southeast_asia", 23.8103, 90.4125),
    Metro("Colombo", "south_southeast_asia", 6.9271, 79.8612),
    Metro("Bangkok", "south_southeast_asia", 13.7563, 100.5018),
    Metro("Kuala Lumpur", "south_southeast_asia", 3.1390, 101.6869),
    Metro("Singapore", "south_southeast_asia", 1.3521, 103.8198),
    Metro("Jakarta", "south_southeast_asia", -6.2088, 106.8456),
    Metro("Surabaya", "south_southeast_asia", -7.2575, 112.7521),
    Metro("Manila", "south_southeast_asia", 14.5995, 120.9842),
    Metro("Ho Chi Minh City", "south_southeast_asia", 10.8231, 106.6297),
    # South America
    Metro("Bogota", "south_america", 4.7110, -74.0721),
    Metro("Quito", "south_america", -0.1807, -78.4678),
    Metro("Lima", "south_america", -12.0464, -77.0428),
    Metro("Santiago", "south_america", -33.4489, -70.6693),
    Metro("Buenos Aires", "south_america", -34.6037, -58.3816),
    Metro("Montevideo", "south_america", -34.9011, -56.1645),
    Metro("Sao Paulo", "south_america", -23.5505, -46.6333),
    Metro("Rio de Janeiro", "south_america", -22.9068, -43.1729),
    Metro("Belo Horizonte", "south_america", -19.9167, -43.9345),
    Metro("Brasilia", "south_america", -15.7939, -47.8828),
    Metro("Fortaleza", "south_america", -3.7319, -38.5267),
    Metro("Recife", "south_america", -8.0476, -34.8770),
    # Africa and Middle East
    Metro("Casablanca", "africa_middle_east", 33.5731, -7.5898),
    Metro("Algiers", "africa_middle_east", 36.7538, 3.0588),
    Metro("Tunis", "africa_middle_east", 36.8065, 10.1815),
    Metro("Lagos", "africa_middle_east", 6.5244, 3.3792),
    Metro("Accra", "africa_middle_east", 5.6037, -0.1870),
    Metro("Cairo", "africa_middle_east", 30.0444, 31.2357),
    Metro("Tel Aviv", "africa_middle_east", 32.0853, 34.7818),
    Metro("Istanbul", "africa_middle_east", 41.0082, 28.9784),
    Metro("Dubai", "africa_middle_east", 25.2048, 55.2708),
    Metro("Riyadh", "africa_middle_east", 24.7136, 46.6753),
    Metro("Nairobi", "africa_middle_east", -1.2921, 36.8219),
    Metro("Kampala", "africa_middle_east", 0.3476, 32.5825),
    Metro("Johannesburg", "africa_middle_east", -26.2041, 28.0473),
    Metro("Durban", "africa_middle_east", -29.8587, 31.0218),
    Metro("Cape Town", "africa_middle_east", -33.9249, 18.4241),
    # Oceania
    Metro("Perth", "oceania", -31.9523, 115.8613),
    Metro("Adelaide", "oceania", -34.9285, 138.6007),
    Metro("Melbourne", "oceania", -37.8136, 144.9631),
    Metro("Sydney", "oceania", -33.8688, 151.2093),
    Metro("Canberra", "oceania", -35.2809, 149.1300),
    Metro("Brisbane", "oceania", -27.4698, 153.0251),
    Metro("Auckland", "oceania", -36.8509, 174.7645),
    Metro("Wellington", "oceania", -41.2866, 174.7756),
)

REGION_WEIGHTS = (
    ("north_america", 0.27),
    ("europe", 0.27),
    ("east_asia", 0.17),
    ("south_southeast_asia", 0.11),
    ("south_america", 0.07),
    ("africa_middle_east", 0.07),
    ("oceania", 0.04),
)

REGION_ADJACENCY = {
    "north_america": ("europe", "south_america"),
    "europe": ("north_america", "africa_middle_east"),
    "east_asia": ("south_southeast_asia", "oceania"),
    "south_southeast_asia": ("east_asia", "oceania", "africa_middle_east"),
    "south_america": ("north_america",),
    "africa_middle_east": ("europe", "south_southeast_asia"),
    "oceania": ("east_asia", "south_southeast_asia"),
}

METROS_BY_REGION: dict[str, tuple[Metro, ...]] = {
    region: tuple(metro for metro in METROS if metro.region == region)
    for region, _ in REGION_WEIGHTS
}


def stable_fraction(value: str) -> float:
    """Return a deterministic value in [0, 1) for an arbitrary string."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stable_index(value: str, size: int) -> int:
    """Return a deterministic index for a non-empty sequence."""

    if size <= 0:
        raise ValueError("stable_index requires a non-empty sequence")
    return min(int(stable_fraction(value) * size), size - 1)


def resolve_compose_path(raw_path: str) -> Path:
    """Resolve an Output directory or a direct Compose file path."""

    path = Path(raw_path).expanduser()
    if path.is_dir():
        candidates = (path / "docker-compose.yml", path / "docker-compose.yaml")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"no docker-compose.yml/yaml found in {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Compose path does not exist: {path}")
    return path.resolve()


def geo_output_path(source: Path, output_template: str | None = None) -> Path:
    """Return a safe sibling output path without ever reusing the source path."""

    if output_template is None:
        name = f"{source.stem}-geo{source.suffix}"
    else:
        directory_name = source.parent.name
        scale = (
            directory_name.removeprefix("output_")
            if directory_name.startswith("output_")
            else directory_name
        )
        try:
            name = output_template.format(
                scale=scale,
                stem=source.stem,
                suffix=source.suffix,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid --output-template: {error}") from error

    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--output-template must produce a filename, not a path")
    output = source.with_name(name)
    if output == source:
        raise ValueError("output filename must differ from the source filename")
    return output


def top_level_section(lines: list[str], section_name: str) -> tuple[int, int]:
    """Return the half-open line range containing a top-level YAML section."""

    start = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.rstrip("\r\n") == f"{section_name}:"
        ),
        None,
    )
    if start is None:
        raise ValueError(f"top-level '{section_name}:' section not found")

    end = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped and not lines[index][0].isspace() and not stripped.startswith("#"):
            end = index
            break
    return start, end


def mapping_ranges(lines: list[str], section_name: str) -> list[tuple[int, int, str]]:
    """Find mapping blocks without parsing and reformatting the YAML document."""

    section_start, section_end = top_level_section(lines, section_name)
    starts: list[tuple[int, str]] = []
    for index in range(section_start, section_end):
        match = SERVICE_RE.match(lines[index].rstrip("\r\n"))
        if match:
            starts.append((index, match.group(1).strip('"\'')))

    ranges: list[tuple[int, int, str]] = []
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else section_end
        ranges.append((start, end, name))
    return ranges


def service_ranges(lines: list[str]) -> list[tuple[int, int, str]]:
    """Find Compose service blocks while retaining original formatting."""

    return mapping_ranges(lines, "services")


def network_ranges(lines: list[str]) -> list[tuple[int, int, str]]:
    """Find top-level Compose network blocks while retaining formatting."""

    return mapping_ranges(lines, "networks")


def unquote_yaml_scalar(value: str) -> str:
    """Decode the simple quoted/unquoted scalar forms emitted by SeedEMU."""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace(r'\"', '"').replace(r"\\", "\\")
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value.split(" #", 1)[0].strip()


def yaml_double_quote(value: str) -> str:
    """Return a safe double-quoted YAML scalar for generated label values."""

    escaped = value.replace("\\", "\\\\").replace('"', r'\"')
    return f'"{escaped}"'


def extract_label(block: Iterable[str], key: str) -> str | None:
    """Extract an exact mapping-style label value from a service/network block."""

    prefix = f"{key}:"
    for line in block:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return unquote_yaml_scalar(stripped[len(prefix) :])
    return None


def extract_asn(block: Iterable[str]) -> str | None:
    """Extract the ASN label from a service block, if present."""

    for line in block:
        match = ASN_RE.match(line)
        if match:
            return match.group(1)
    return None


def extract_service_networks(block: list[str]) -> list[str]:
    """Return Compose network keys attached to a service mapping."""

    network_names: list[str] = []
    in_networks = False
    for line in block:
        raw = line.rstrip("\r\n")
        if raw == "        networks:":
            in_networks = True
            continue
        if not in_networks:
            continue

        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indentation = len(raw) - len(raw.lstrip())
        if indentation <= 8:
            break
        if match := SERVICE_NETWORK_RE.match(raw):
            network_names.append(match.group(1).strip('"\''))
        elif match := SERVICE_NETWORK_LIST_RE.match(raw):
            network_names.append(match.group(1).strip('"\''))
    return network_names


def asn_sort_key(asn: str) -> tuple[int, int | str]:
    """Sort numeric ASNs numerically and retain support for text ASNs."""

    return (0, int(asn)) if asn.isdigit() else (1, asn)


def parse_compose_topology(text: str) -> ComposeTopology:
    """Extract services and network types from Compose text."""

    lines = text.splitlines(keepends=True)
    network_types = {
        name: extract_label(lines[start:end], NETWORK_TYPE_KEY) or ""
        for start, end, name in network_ranges(lines)
    }
    services: dict[str, ServiceRecord] = {}
    for start, end, name in service_ranges(lines):
        block = lines[start:end]
        asn = extract_asn(block)
        if asn is None:
            continue
        services[name] = ServiceRecord(name, asn, tuple(extract_service_networks(block)))
    return ComposeTopology(services, network_types)


def great_circle_distance_km(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Return haversine distance between two latitude/longitude coordinates."""

    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1
    a = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def spherical_centroid(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Calculate a latitude/longitude centroid on the unit sphere."""

    if not points:
        return None

    x = y = z = 0.0
    for latitude, longitude in points:
        latitude_radians = math.radians(latitude)
        longitude_radians = math.radians(longitude)
        x += math.cos(latitude_radians) * math.cos(longitude_radians)
        y += math.cos(latitude_radians) * math.sin(longitude_radians)
        z += math.sin(latitude_radians)

    horizontal = math.hypot(x, y)
    if math.hypot(horizontal, z) < 1e-12:
        return None
    return math.degrees(math.atan2(z, horizontal)), math.degrees(math.atan2(y, x))


def spherical_medoid(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Return the member point minimizing total great-circle distance."""

    if not points:
        return None
    if len(points) == 1:
        return points[0]
    best_index = min(
        range(len(points)),
        key=lambda index: (
            sum(
                great_circle_distance_km(points[index], candidate)
                for candidate in points
            ),
            index,
        ),
    )
    return points[best_index]


def select_home_region(asn: str, seed: str) -> str:
    """Assign an AS to a macro region using stable Internet-density weights."""

    target = stable_fraction(f"{seed}:region:{asn}")
    cumulative = 0.0
    for region, weight in REGION_WEIGHTS:
        cumulative += weight
        if target < cumulative:
            return region
    return REGION_WEIGHTS[-1][0]


def pop_count_for_size(node_count: int, spread_factor: float = 1.0) -> int:
    """Choose the number of PoPs for an AS size tier."""

    if node_count <= 0:
        return 0
    if node_count == 1:
        base = 1
    elif node_count <= 4:
        base = 2
    elif node_count <= 9:
        base = 3
    elif node_count <= 19:
        base = 5
    elif node_count <= 49:
        base = 8
    else:
        base = 12
    scaled = 1 + round((base - 1) * spread_factor)
    return min(node_count, max(1, scaled))


def footprint_km_for_size(node_count: int, spread_factor: float = 1.0) -> float:
    """Return the intended AS radius from its home metro for a size tier."""

    if node_count <= 1:
        base = 0.0
    elif node_count <= 4:
        base = 1000.0
    elif node_count <= 9:
        base = 2200.0
    elif node_count <= 19:
        base = 3500.0
    elif node_count <= 49:
        base = 5500.0
    else:
        base = 9000.0
    return base * spread_factor


def select_as_metros(
    asn: str,
    node_count: int,
    seed: str,
    spread_factor: float,
) -> tuple[Metro, ...]:
    """Select a coherent set of city PoPs for one AS."""

    region = select_home_region(asn, seed)
    regional_metros = list(METROS_BY_REGION[region])
    home = regional_metros[stable_index(f"{seed}:home:{asn}", len(regional_metros))]
    desired = min(
        pop_count_for_size(node_count, spread_factor), len(regional_metros) + 4
    )
    if desired == 1:
        return (home,)

    budget = footprint_km_for_size(node_count, spread_factor)
    ranked_home_region = sorted(
        regional_metros,
        key=lambda metro: (
            great_circle_distance_km(
                (home.latitude, home.longitude), (metro.latitude, metro.longitude)
            ),
            metro.name,
        ),
    )
    candidates = [
        metro
        for metro in ranked_home_region
        if great_circle_distance_km(
            (home.latitude, home.longitude), (metro.latitude, metro.longitude)
        )
        <= budget
    ]

    # Sparse regions still need the promised number of PoPs.  Nearest same-region
    # cities are preferred, so this relaxation remains geographically coherent.
    for metro in ranked_home_region:
        if len(candidates) >= desired:
            break
        if metro not in candidates:
            candidates.append(metro)

    # Only the largest backbones may use one adjacent region.  Restricting the
    # extension to four closest cities creates a corridor rather than scattering
    # nodes across unrelated continents.
    if node_count >= 50 and desired >= 10:
        adjacent_regions = REGION_ADJACENCY[region]
        adjacent_region = adjacent_regions[
            stable_index(f"{seed}:adjacent-region:{asn}", len(adjacent_regions))
        ]
        adjacent_ranked = sorted(
            METROS_BY_REGION[adjacent_region],
            key=lambda metro: (
                great_circle_distance_km(
                    (home.latitude, home.longitude),
                    (metro.latitude, metro.longitude),
                ),
                metro.name,
            ),
        )
        candidates.extend(adjacent_ranked[:4])

    # Farthest-point selection fills the available geographic footprint and is
    # paired with graph-farthest seed nodes below.
    selected = [home]
    while len(selected) < desired:
        remaining = [metro for metro in candidates if metro not in selected]
        if not remaining:
            break
        chosen = max(
            remaining,
            key=lambda metro: (
                min(
                    great_circle_distance_km(
                        (metro.latitude, metro.longitude),
                        (existing.latitude, existing.longitude),
                    )
                    for existing in selected
                ),
                stable_fraction(f"{seed}:metro-tie:{asn}:{metro.name}"),
            ),
        )
        selected.append(chosen)
    return tuple(selected)


def breadth_first_distances(
    adjacency: dict[str, set[str]], start: str
) -> dict[str, int]:
    """Return unweighted graph distances from one node."""

    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def choose_graph_seeds(
    nodes: set[str],
    adjacency: dict[str, set[str]],
    count: int,
    asn: str,
    seed: str,
) -> tuple[str, ...]:
    """Choose topologically separated service nodes as PoP representatives."""

    ordered = sorted(nodes)
    if not ordered or count <= 0:
        return ()
    first = ordered[stable_index(f"{seed}:first-node:{asn}", len(ordered))]
    selected = [first]
    minimum_distances = {node: math.inf for node in ordered}

    while len(selected) < min(count, len(ordered)):
        latest_distances = breadth_first_distances(adjacency, selected[-1])
        for node in ordered:
            minimum_distances[node] = min(
                minimum_distances[node], latest_distances.get(node, math.inf)
            )
        remaining = [node for node in ordered if node not in selected]
        selected.append(
            max(
                remaining,
                key=lambda node: (
                    minimum_distances[node],
                    stable_fraction(f"{seed}:graph-tie:{asn}:{node}"),
                ),
            )
        )
    return tuple(selected)


def assign_nodes_to_seeds(
    nodes: set[str],
    adjacency: dict[str, set[str]],
    seeds: tuple[str, ...],
    asn: str,
    seed: str,
) -> dict[str, int]:
    """Assign each node to its closest graph seed with deterministic ties."""

    distance_maps = [breadth_first_distances(adjacency, item) for item in seeds]
    assignments: dict[str, int] = {}
    for node in sorted(nodes):
        distances = [mapping.get(node, math.inf) for mapping in distance_maps]
        minimum = min(distances)
        choices = [index for index, distance in enumerate(distances) if distance == minimum]
        if math.isinf(minimum):
            choices = list(range(len(seeds)))
        choice = choices[
            stable_index(f"{seed}:assign:{asn}:{node}", len(choices))
        ]
        assignments[node] = choice
    return assignments


def offset_coordinate(
    metro: Metro, key: str, radius_degrees: float
) -> tuple[float, float]:
    """Add a stable radial offset while keeping longitude distance sensible."""

    if radius_degrees == 0:
        return metro.latitude, metro.longitude
    angle = 2 * math.pi * stable_fraction(f"{key}:angle")
    radius = radius_degrees * math.sqrt(stable_fraction(f"{key}:radius"))
    latitude = metro.latitude + radius * math.sin(angle)
    longitude_scale = max(0.35, math.cos(math.radians(metro.latitude)))
    longitude = metro.longitude + radius * math.cos(angle) / longitude_scale
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    return max(-85.0, min(85.0, latitude)), longitude


def build_geographic_layout(
    compose_texts: dict[Path, str],
    seed: str,
    spread_factor: float = 1.0,
    local_jitter: float = 0.12,
) -> GeographicLayout:
    """Build one deterministic topology-aware layout across all input files."""

    topologies = {
        path: parse_compose_topology(text) for path, text in compose_texts.items()
    }
    as_nodes: defaultdict[str, set[str]] = defaultdict(set)
    local_members: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for topology in topologies.values():
        for record in topology.services.values():
            as_nodes[record.asn].add(record.name)
            for network_name in record.networks:
                if topology.network_types.get(network_name) != "global":
                    local_members[(record.asn, network_name)].add(record.name)

    adjacency: dict[str, dict[str, set[str]]] = {
        asn: {node: set() for node in nodes} for asn, nodes in as_nodes.items()
    }
    for (asn, _), members in local_members.items():
        ordered = sorted(members)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                adjacency[asn][first].add(second)
                adjacency[asn][second].add(first)

    placements: dict[tuple[str, str], NodePlacement] = {}
    as_regions: dict[str, tuple[str, ...]] = {}
    as_pops: dict[str, tuple[Metro, ...]] = {}
    for asn in sorted(as_nodes, key=asn_sort_key):
        nodes = as_nodes[asn]
        metros = select_as_metros(asn, len(nodes), seed, spread_factor)
        graph_seeds = choose_graph_seeds(
            nodes, adjacency[asn], len(metros), asn, seed
        )
        assignments = assign_nodes_to_seeds(
            nodes, adjacency[asn], graph_seeds, asn, seed
        )
        as_pops[asn] = metros
        as_regions[asn] = tuple(dict.fromkeys(metro.region for metro in metros))
        for node in sorted(nodes):
            metro = metros[assignments[node]]
            latitude, longitude = offset_coordinate(
                metro,
                f"{seed}:node:{asn}:{node}",
                local_jitter,
            )
            placements[(asn, node)] = NodePlacement(
                latitude, longitude, metro.name, metro.region
            )

    return GeographicLayout(placements, as_regions, as_pops, adjacency)


def nearest_metro(point: tuple[float, float]) -> Metro:
    """Return the city anchor nearest to a coordinate."""

    return min(
        METROS,
        key=lambda metro: (
            great_circle_distance_km(
                point, (metro.latitude, metro.longitude)
            ),
            metro.name,
        ),
    )


def update_service_block(
    block: list[str],
    service_name: str,
    asn: str,
    placement: NodePlacement,
    newline: str,
) -> list[str]:
    """Replace or insert the four geo labels in one service block."""

    cleaned = [line for line in block if not GEO_RE.match(line)]
    insert_after = None
    for index, line in enumerate(cleaned):
        if re.match(rf"^\s+{re.escape(ROLE_KEY)}:", line):
            insert_after = index
            break
    if insert_after is None:
        for index, line in enumerate(cleaned):
            if ASN_RE.match(line):
                insert_after = index
                break
    if insert_after is None:
        raise ValueError(f"service {service_name!r} lost its ASN label during update")

    geo_lines = [
        f'            {GEO_PREFIX}lat: "{placement.latitude:.6f}"{newline}',
        f'            {GEO_PREFIX}lon: "{placement.longitude:.6f}"{newline}',
        f'            {GEO_PREFIX}source: "generated-hierarchical-v2"{newline}',
        f"            {GEO_PREFIX}label: "
        f"{yaml_double_quote(f'AS{asn} / {placement.metro}')}{newline}",
    ]
    cleaned[insert_after + 1 : insert_after + 1] = geo_lines
    return cleaned


def update_network_block(
    block: list[str],
    network_name: str,
    member_points: list[tuple[float, float]],
    seed: str,
    newline: str,
) -> tuple[list[str], bool]:
    """Place a global network near the city closest to its member medoid."""

    cleaned = [line for line in block if not GEO_RE.match(line)]
    medoid = spherical_medoid(member_points)
    used_fallback = medoid is None
    if medoid is None:
        metro = METROS[stable_index(f"{seed}:network:{network_name}", len(METROS))]
    else:
        metro = nearest_metro(medoid)
    latitude, longitude = offset_coordinate(
        metro, f"{seed}:ix:{network_name}", 0.08
    )

    label = extract_label(cleaned, NETWORK_DISPLAYNAME_KEY)
    if not label:
        scope = extract_label(cleaned, NETWORK_SCOPE_KEY)
        metadata_name = extract_label(cleaned, NETWORK_NAME_KEY)
        label = "/".join(item for item in (scope, metadata_name) if item) or network_name

    insert_after = None
    for key in (
        NETWORK_DISPLAYNAME_KEY,
        NETWORK_PREFIX_KEY,
        NETWORK_NAME_KEY,
        NETWORK_SCOPE_KEY,
        NETWORK_TYPE_KEY,
    ):
        for index, line in enumerate(cleaned):
            if line.strip().startswith(f"{key}:"):
                insert_after = index
                break
        if insert_after is not None:
            break
    if insert_after is None:
        raise ValueError(f"global network {network_name!r} has no metadata insertion point")

    source = (
        "generated-ix-fallback-v2" if used_fallback else "generated-ix-medoid-v2"
    )
    geo_lines = [
        f'            {GEO_PREFIX}lat: "{latitude:.6f}"{newline}',
        f'            {GEO_PREFIX}lon: "{longitude:.6f}"{newline}',
        f"            {GEO_PREFIX}source: {yaml_double_quote(source)}{newline}",
        f"            {GEO_PREFIX}label: "
        f"{yaml_double_quote(f'{label} / {metro.name}')}{newline}",
    ]
    cleaned[insert_after + 1 : insert_after + 1] = geo_lines
    return cleaned, used_fallback


def update_compose(
    text: str,
    placements: dict[tuple[str, str], NodePlacement],
    seed: str,
) -> tuple[str, int, int, int, int]:
    """Update service/global-network geo and return text plus coverage counts."""

    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    ranges = service_ranges(lines)
    changed = 0
    skipped = 0
    network_members: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)

    for start, end, service_name in ranges:
        block = lines[start:end]
        asn = extract_asn(block)
        if asn is None:
            continue
        placement = placements.get((asn, service_name))
        if placement is None:
            raise ValueError(f"no generated placement for AS{asn} service {service_name}")
        point = (placement.latitude, placement.longitude)
        for network_name in extract_service_networks(block):
            network_members[network_name].append(point)

    # Work backwards so earlier line indexes remain valid after insertion.
    for start, end, service_name in reversed(ranges):
        block = lines[start:end]
        asn = extract_asn(block)
        if asn is None:
            skipped += 1
            continue
        lines[start:end] = update_service_block(
            block, service_name, asn, placements[(asn, service_name)], newline
        )
        changed += 1

    global_networks = 0
    fallback_networks = 0
    for start, end, network_name in reversed(network_ranges(lines)):
        block = lines[start:end]
        if extract_label(block, NETWORK_TYPE_KEY) != "global":
            continue
        lines[start:end], used_fallback = update_network_block(
            block, network_name, network_members[network_name], seed, newline
        )
        global_networks += 1
        fallback_networks += int(used_fallback)
    return "".join(lines), changed, skipped, global_networks, fallback_networks


def atomic_write(path: Path, text: str, source_mode: int) -> None:
    """Atomically create/replace an output file with predictable permissions."""

    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else source_mode
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create geo-labelled Compose copies using deterministic, topology-aware "
            "real-city AS PoPs; source files are never modified."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Output directories or docker-compose.yml/yaml files",
    )
    parser.add_argument(
        "--seed",
        default="seedemu-geo-hierarchical-v2",
        help="stable seed used for regions, PoPs, and offsets (default: %(default)s)",
    )
    parser.add_argument(
        "--spread-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "scale AS PoP counts and geographic footprints; valid range 0.5..2.0 "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--local-jitter",
        "--cluster-radius",
        dest="local_jitter",
        type=float,
        default=0.12,
        metavar="DEGREES",
        help=(
            "maximum node offset around its city PoP; --cluster-radius remains a "
            "backward-compatible alias (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-template",
        metavar="TEMPLATE",
        help=(
            "optional sibling filename template; supports {scale}, {stem}, and {suffix} "
            "(example: '{scale}geo.yaml')"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report output files without writing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.5 <= args.spread_factor <= 2.0:
        raise SystemExit("--spread-factor must be between 0.5 and 2.0")
    if not 0 <= args.local_jitter < 0.5:
        raise SystemExit("--local-jitter must be at least 0 and less than 0.5 degrees")

    try:
        paths = list(dict.fromkeys(resolve_compose_path(raw) for raw in args.paths))
        compose_texts = {path: path.read_text(encoding="utf-8") for path in paths}
        layout = build_geographic_layout(
            compose_texts, args.seed, args.spread_factor, args.local_jitter
        )
        if not layout.placements:
            raise ValueError("no services with SeedEMU ASN labels were found")
        outputs = {
            source: geo_output_path(source, args.output_template) for source in paths
        }
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("--output-template maps multiple sources to the same output")

        for source, original in compose_texts.items():
            result, targeted, skipped, global_networks, fallback_networks = update_compose(
                original, layout.placements, args.seed
            )
            output = outputs[source]
            existing = output.read_text(encoding="utf-8") if output.exists() else None
            if existing == result:
                action = "unchanged"
            elif args.dry_run:
                action = "would create" if existing is None else "would update"
            else:
                source_mode = stat.S_IMODE(source.stat().st_mode)
                atomic_write(output, result, source_mode)
                action = "created" if existing is None else "updated"
            print(
                f"{action} {output} from {source}: {targeted} ASN services, "
                f"{skipped} services without ASN skipped, {global_networks} global networks "
                f"({fallback_networks} fallback)"
            )

        region_counts: defaultdict[str, int] = defaultdict(int)
        for regions in layout.as_regions.values():
            region_counts[regions[0]] += 1
        pop_total = sum(len(pops) for pops in layout.as_pops.values())
        distribution = ", ".join(
            f"{region}={region_counts[region]}" for region, _ in REGION_WEIGHTS
        )
        print(
            f"AS layout: {len(layout.as_pops)} ASes, {pop_total} city PoPs, "
            f"spread-factor={args.spread_factor:.2f}, local-jitter="
            f"{args.local_jitter:.3f} degrees, seed={args.seed!r}"
        )
        print(f"Home-region distribution: {distribution}")
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
