#!/usr/bin/env python3
"""Focused tests for hierarchical, topology-aware geo generation."""

from pathlib import Path
import unittest

import add_geo_labels as geo


COMPOSE_FIXTURE = """services:
    router_a:
        networks:
            net_ix:
                ipv4_address: 10.0.0.2
            net_local:
                ipv4_address: 10.1.0.2
        labels:
            org.seedsecuritylabs.seedemu.meta.asn: "100"
            org.seedsecuritylabs.seedemu.meta.role: "BorderRouter"
            org.seedsecuritylabs.seedemu.meta.net.0.name: "ix1"
            org.seedsecuritylabs.seedemu.meta.net.0.ifname: "eth0"
            org.seedsecuritylabs.seedemu.meta.net.0.networkname: "net_ix"
            org.seedsecuritylabs.seedemu.meta.net.0.networkid: "runtime-value"

    router_b:
        networks:
            net_ix:
                ipv4_address: 10.0.0.3
        labels:
            org.seedsecuritylabs.seedemu.meta.asn: "200"
            org.seedsecuritylabs.seedemu.meta.role: "BorderRouter"

networks:
    net_ix:
        labels:
            org.seedsecuritylabs.seedemu.meta.type: "global"
            org.seedsecuritylabs.seedemu.meta.scope: "ix"
            org.seedsecuritylabs.seedemu.meta.name: "ix1"
            org.seedsecuritylabs.seedemu.meta.displayname: "IX One"

    net_local:
        labels:
            org.seedsecuritylabs.seedemu.meta.type: "local"
            org.seedsecuritylabs.seedemu.meta.scope: "100"
            org.seedsecuritylabs.seedemu.meta.name: "local"
"""


def chain_compose(node_count: int, asn: str = "100") -> str:
    """Build a small line topology whose graph partitions are easy to inspect."""

    service_blocks = []
    for index in range(node_count):
        attached = []
        if index > 0:
            attached.append(f"net_{index - 1}_{index}")
        if index + 1 < node_count:
            attached.append(f"net_{index}_{index + 1}")
        networks = "\n".join(
            f"            {name}:\n                ipv4_address: 10.0.{index}.2"
            for name in attached
        )
        service_blocks.append(
            f"""    router_{index}:
        networks:
{networks}
        labels:
            org.seedsecuritylabs.seedemu.meta.asn: "{asn}"
            org.seedsecuritylabs.seedemu.meta.role: "Router"
"""
        )

    network_blocks = []
    for index in range(node_count - 1):
        network_blocks.append(
            f"""    net_{index}_{index + 1}:
        labels:
            org.seedsecuritylabs.seedemu.meta.type: "local"
            org.seedsecuritylabs.seedemu.meta.scope: "{asn}"
            org.seedsecuritylabs.seedemu.meta.name: "link_{index}_{index + 1}"
"""
        )
    return "services:\n" + "\n".join(service_blocks) + "networks:\n" + "\n".join(
        network_blocks
    )


class GeoGenerationTests(unittest.TestCase):
    def test_services_and_global_network_receive_geo(self) -> None:
        inputs = {Path("fixture.yml"): COMPOSE_FIXTURE}
        layout = geo.build_geographic_layout(inputs, "test-seed")
        result, services, skipped, global_networks, fallbacks = geo.update_compose(
            COMPOSE_FIXTURE, layout.placements, "test-seed"
        )

        self.assertEqual((services, skipped, global_networks, fallbacks), (2, 0, 1, 0))
        self.assertEqual(result.count(f"{geo.GEO_PREFIX}lat:"), 3)
        self.assertIn(f'{geo.GEO_PREFIX}source: "generated-ix-medoid-v2"', result)
        self.assertIn(f'{geo.GEO_PREFIX}source: "generated-hierarchical-v2"', result)
        self.assertIn(f'{geo.GEO_PREFIX}label: "IX One / ', result)
        self.assertIn('meta.net.0.ifname: "eth0"', result)
        self.assertIn('meta.net.0.networkname: "net_ix"', result)
        self.assertIn('meta.net.0.networkid: "runtime-value"', result)

        local_block = next(
            result.splitlines(keepends=True)[start:end]
            for start, end, name in geo.network_ranges(result.splitlines(keepends=True))
            if name == "net_local"
        )
        self.assertFalse(any(geo.GEO_RE.match(line) for line in local_block))

    def test_size_tiers_expand_pop_count_and_footprint(self) -> None:
        sizes = (1, 4, 9, 19, 49, 50)
        self.assertEqual(
            [geo.pop_count_for_size(size) for size in sizes],
            [1, 2, 3, 5, 8, 12],
        )
        footprints = [geo.footprint_km_for_size(size) for size in sizes]
        self.assertEqual(footprints, sorted(footprints))
        self.assertGreater(footprints[-1], footprints[1] * 5)

    def test_chain_layout_is_spread_coherent_and_reproducible(self) -> None:
        compose = chain_compose(9)
        inputs = {Path("chain.yml"): compose}
        first = geo.build_geographic_layout(inputs, "chain-seed")
        second = geo.build_geographic_layout(inputs, "chain-seed")

        self.assertEqual(first.placements, second.placements)
        self.assertEqual(len(first.as_pops["100"]), 3)
        self.assertEqual(len(first.as_regions["100"]), 1)
        self.assertEqual(
            len({placement.metro for placement in first.placements.values()}), 3
        )

        points = {
            name: (
                first.placements[("100", name)].latitude,
                first.placements[("100", name)].longitude,
            )
            for name in first.adjacency["100"]
        }
        pair_distances = [
            geo.great_circle_distance_km(points[first_name], points[second_name])
            for position, first_name in enumerate(sorted(points))
            for second_name in sorted(points)[position + 1 :]
        ]
        edge_distances = [
            geo.great_circle_distance_km(points[node], points[neighbor])
            for node, neighbors in first.adjacency["100"].items()
            for neighbor in neighbors
            if node < neighbor
        ]
        self.assertGreater(max(pair_distances), 250.0)
        self.assertLessEqual(
            sum(edge_distances) / len(edge_distances),
            sum(pair_distances) / len(pair_distances),
        )

    def test_large_backbone_uses_at_most_one_adjacent_region(self) -> None:
        metros = geo.select_as_metros("64500", 86, "large-seed", 1.0)
        regions = tuple(dict.fromkeys(metro.region for metro in metros))
        self.assertLessEqual(len(regions), 2)
        if len(regions) == 2:
            self.assertIn(regions[1], geo.REGION_ADJACENCY[regions[0]])

    def test_offsets_stay_near_real_city_anchor(self) -> None:
        metro = next(item for item in geo.METROS if item.name == "New York")
        point = geo.offset_coordinate(metro, "sample", 0.12)
        self.assertLess(
            geo.great_circle_distance_km(
                point, (metro.latitude, metro.longitude)
            ),
            20.0,
        )

    def test_spherical_centroid_handles_dateline(self) -> None:
        latitude, longitude = geo.spherical_centroid([(0.0, 179.0), (0.0, -179.0)])
        self.assertAlmostEqual(latitude, 0.0)
        self.assertAlmostEqual(abs(longitude), 180.0)

    def test_scale_output_template(self) -> None:
        source = Path("/tmp/output_3083/docker-compose.yml")
        self.assertEqual(
            geo.geo_output_path(source, "{scale}geo.yaml").name,
            "3083geo.yaml",
        )


if __name__ == "__main__":
    unittest.main()
