"""Extract yearly stock minute ZIP files into year-scoped folders."""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "ashare_minute_data_2023_2026_04"


def safe_member_name(name: str, year: str) -> Path | None:
    parts = Path(name).parts
    if not parts or name.endswith("/"):
        return None
    if parts[0] == year:
        parts = parts[1:]
    if not parts:
        return None
    rel = Path(*parts)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe zip member path: {name}")
    return rel


def extract_zip(zip_path: Path, output_root: Path, force: bool = False) -> list[dict[str, str | int]]:
    year = zip_path.stem
    year_dir = output_root / year
    year_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            rel = safe_member_name(info.filename, year)
            if rel is None:
                continue
            target = year_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            status = "extracted"
            if target.exists() and not force and target.stat().st_size == info.file_size:
                status = "skipped_existing"
            else:
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            rows.append(
                {
                    "zip": zip_path.name,
                    "year": year,
                    "member": info.filename,
                    "output_path": str(target),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "status": status,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite files even when size matches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.csv"

    all_rows: list[dict[str, str | int]] = []
    for zip_path in sorted(data_dir.glob("*.zip")):
        print(f"Extracting {zip_path.name} -> {output_dir / zip_path.stem}")
        rows = extract_zip(zip_path, output_dir, force=args.force)
        all_rows.extend(rows)
        extracted = sum(1 for row in rows if row["status"] == "extracted")
        skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
        print(f"  files={len(rows)} extracted={extracted} skipped={skipped}")

    if all_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Manifest saved to {manifest_path}")
    else:
        print(f"No ZIP files found in {data_dir}")


if __name__ == "__main__":
    main()
