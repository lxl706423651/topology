#!/usr/bin/env python3
"""Build SeedEMU images with patched BIRD baked into fast-build base images.

This wrapper intentionally does not modify 1buildFastDockerImages.py. It creates
a temporary copy of the compiler output, rewrites only that copy, and then calls
the original fast archive helper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_fast_builder(path: Path):
    spec = importlib.util.spec_from_file_location("seedemu_fast_docker_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fast builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_compose(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("services"), dict):
        raise RuntimeError(f"invalid compose file: {path}")
    return data


def copy_output_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)

    ignored = {".fastDockerBuild", ".deltaFastDockerBuild", "run_logs", "summary.json"}

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored}

    shutil.copytree(source, dest, ignore=ignore)


def first_from_image(dockerfile: Path) -> str | None:
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^FROM\s+(\S+)", stripped, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    return None


def rewrite_first_from(dockerfile: Path, new_image: str) -> None:
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^\s*FROM\s+", line, re.IGNORECASE):
            lines[idx] = re.sub(r"^(\s*FROM\s+)\S+", rf"\1{new_image}", line, count=1, flags=re.IGNORECASE)
            dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise RuntimeError(f"missing FROM in {dockerfile}")


def patch_bird_conf_in_context(dockerfile: Path) -> bool:
    patched = False
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*COPY\s+(\S+)\s+/etc/bird/bird\.conf\s*$", line)
        if not match:
            continue
        src = dockerfile.parent / match.group(1)
        if not src.is_file():
            raise RuntimeError(f"missing bird.conf COPY source: {src}")
        text = src.read_text(encoding="utf-8")
        include = 'include "/etc/bird/conf/delta.conf";'
        explicit_re = re.compile(r'^\s*include\s+"/etc/bird/conf/delta\.conf";\s*$')
        wildcard_re = re.compile(r'^\s*include\s+"/etc/bird/conf/\*\.conf";\s*$')
        lines = [item for item in text.splitlines() if not explicit_re.match(item)]
        has_wildcard = any(wildcard_re.match(item) for item in lines)
        new_text = "\n".join(lines).rstrip() + "\n"
        if not has_wildcard:
            new_text = new_text.rstrip() + "\n\n" + include + "\n"
        if new_text != text:
            src.write_text(new_text, encoding="utf-8")
        patched = True
    return patched


def build_delta_base(
    *,
    source_image: str,
    fast_builder,
    by_tag: dict,
    base_root: Path,
    bird_binary: Path,
    agent: Path,
    metric: int,
) -> str:
    digest = sha256_text(f"{source_image}\n{bird_binary.read_bytes().hex()}\n{agent.read_bytes().hex()}\n{metric}\n")[:24]
    tag = f"seedemu-delta-router-base:{digest}"
    if fast_builder.checkDockerImageExists(tag):
        return tag

    fast_builder.ensureImageAvailable(source_image, by_tag)
    context = base_root / digest
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    shutil.copy2(bird_binary, context / "bird")
    shutil.copy2(agent, context / "delta_fib_agent.py")
    (context / "delta.conf").write_text(
        "\n".join(
            [
                "protocol delta delta_main {",
                '    event "/run/bird/delta-events.tsv";',
                f"    metric {metric};",
                "    merge paths on limit 16;",
                "    ipv4 {",
                "        table master4;",
                "        export all;",
                "    };",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (context / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {source_image}",
                "COPY bird /usr/sbin/bird",
                "COPY delta_fib_agent.py /usr/local/bin/delta_fib_agent.py",
                "COPY delta.conf /etc/bird/conf/delta.conf",
                "RUN chmod 0755 /usr/sbin/bird /usr/local/bin/delta_fib_agent.py && mkdir -p /run/bird /etc/bird/conf",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run(["docker", "build", "-t", tag, str(context)])
    return tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compose_file", type=Path)
    parser.add_argument("--fast-builder", type=Path, default=Path("1buildFastDockerImages.py"))
    parser.add_argument("--bird-binary", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--skip-service", action="append", default=[])
    parser.add_argument("--skip-load", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    compose_file = args.compose_file.expanduser().resolve()
    output_dir = compose_file.parent
    work_dir = args.work_dir.expanduser().resolve()
    temp_output = work_dir / "delta_output"
    base_root = work_dir / "delta_base_contexts"
    archive_file = work_dir / "images_all.tar"
    summary_file = work_dir / "summary.json"
    build_images_file = work_dir / "build_images.sh"

    fast_builder_path = args.fast_builder.expanduser().resolve()
    fast_builder = load_fast_builder(fast_builder_path)
    bird_binary = args.bird_binary.expanduser().resolve()
    agent = args.agent.expanduser().resolve()
    if not bird_binary.is_file():
        raise RuntimeError(f"missing patched bird binary: {bird_binary}")
    if not agent.is_file():
        raise RuntimeError(f"missing delta agent: {agent}")

    work_dir.mkdir(parents=True, exist_ok=True)
    base_root.mkdir(parents=True, exist_ok=True)
    copy_output_tree(output_dir, temp_output)
    temp_compose = temp_output / compose_file.name

    os.environ["COMPOSE_PROJECT_NAME"] = args.project_name
    _project, _entries, by_tag = fast_builder.discoverBuildEntries(temp_compose, args.skip_service)

    from_images = sorted({img for df in temp_output.rglob("Dockerfile") if (img := first_from_image(df))})
    base_map = {}
    for source_image in from_images:
        base_map[source_image] = build_delta_base(
            source_image=source_image,
            fast_builder=fast_builder,
            by_tag=by_tag,
            base_root=base_root,
            bird_binary=bird_binary,
            agent=agent,
            metric=args.metric,
        )

    dockerfiles_patched = 0
    bird_confs_patched = 0
    for dockerfile in temp_output.rglob("Dockerfile"):
        source = first_from_image(dockerfile)
        if source and source in base_map:
            rewrite_first_from(dockerfile, base_map[source])
            dockerfiles_patched += 1
        if patch_bird_conf_in_context(dockerfile):
            bird_confs_patched += 1

    fast_summary = fast_builder.createFastArchive(
        compose_file=temp_compose,
        archive_file=archive_file,
        summary_file=summary_file,
        build_images_file=build_images_file,
        cache_dir=work_dir / "cache",
        skip_services=args.skip_service,
    )
    if not args.skip_load:
        run(["docker", "load", "-i", str(archive_file)])

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["deltaBase"] = {
        "status": "PASS",
        "sourceComposeFile": str(compose_file),
        "temporaryComposeFile": str(temp_compose),
        "projectName": args.project_name,
        "metric": args.metric,
        "fastBuilder": str(fast_builder_path),
        "birdBinary": str(bird_binary),
        "agent": str(agent),
        "baseMap": base_map,
        "dockerfilesPatched": dockerfiles_patched,
        "birdConfsPatched": bird_confs_patched,
        "durationSeconds": round(time.monotonic() - started, 2),
        "fastSummary": {
            "imageCount": fast_summary.get("imageCount"),
            "normalImageCount": fast_summary.get("normalImageCount"),
            "baseCount": fast_summary.get("baseCount"),
            "archiveBytes": fast_summary.get("archiveBytes"),
            "durationSeconds": fast_summary.get("durationSeconds"),
        },
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["deltaBase"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
