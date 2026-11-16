#!/usr/bin/env python3
#
# Purpose: convert SeedEMU per-node build compose into a runtime compose that
# uses one shared image and bind-mounts each node's generated config files.
# Inputs: output/docker-compose.yml and generated node directories.
# Outputs: output/docker-compose-fast.yml.
# Side effects: writes only the requested compose file.
# Expected context: run from fullMeshExample after compileFullMesh.sh.

import argparse
from pathlib import Path

import yaml


NODE_FILE_MAP = {
    "082b96ec819c95ae773daebde675ef80": "/start.sh",
    "d18858afc6bb66ec3a19d872077acfd2": "/seedemu_sniffer",
    "17ac2d812a99a91e7f747e1defb72a29": "/seedemu_worker",
    "2b0ae038330eccd43095538618caee7d": "/etc/bird/bird.conf",
    "e01e36443f9f72c6204189260d0bd276": "/ifinfo.txt",
    "d3d51fdf7f4bad30dc5db560a01ce629": "/interface_setup",
    "8e9a384ab59c7626a3ffab63886cc30f": "/etc/bird/conf/kernel.conf",
}


def parseArgs():
    """Parse CLI arguments for fast compose preparation."""
    parser = argparse.ArgumentParser(description="Prepare shared-image Docker compose for SeedEMU output.")
    parser.add_argument("--input", default="output/docker-compose.yml", help="Original compose file.")
    parser.add_argument("--output", default="output/docker-compose-fast.yml", help="Fast compose output file.")
    parser.add_argument("--image", default="seedemu-fullmesh-node:1078", help="Shared runtime image name.")
    return parser.parse_args()


def serviceBuildContext(service):
    """Return service build context when it points at a generated node directory."""
    build = service.get("build")
    if isinstance(build, str):
        return build
    if isinstance(build, dict):
        return build.get("context")
    return None


def isGeneratedNodeService(name, service):
    """Return True for brdnode/rs services that have generated node directories."""
    return (name.startswith("brdnode_") or name.startswith("rs_ix_")) and serviceBuildContext(service)


def createVolumeList(context):
    """Create bind mounts from generated node files into the shared image."""
    context_text = str(context)
    if context_text.startswith("./"):
        context_text = context_text[2:]
    context_path = Path(context_text)
    return [f"./{(context_path / source).as_posix()}:{target}" for source, target in NODE_FILE_MAP.items()]


def main():
    """Write a fast compose file using one shared runtime image."""
    args = parseArgs()
    input_path = Path(args.input)
    output_path = Path(args.output)
    compose = yaml.safe_load(input_path.read_text(encoding="utf-8"))

    new_services = {}
    converted = 0
    removed = 0
    for name, service in compose.get("services", {}).items():
        if not isGeneratedNodeService(name, service):
            removed += 1
            continue

        context = serviceBuildContext(service)
        service = dict(service)
        service.pop("build", None)
        service.pop("depends_on", None)
        service["image"] = args.image
        service["command"] = ["/bin/bash", "/start.sh"]
        service["volumes"] = createVolumeList(context)
        new_services[name] = service
        converted += 1

    compose["services"] = new_services
    output_path.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"converted node services: {converted}; removed non-node services: {removed}; image: {args.image}")


if __name__ == "__main__":
    main()
