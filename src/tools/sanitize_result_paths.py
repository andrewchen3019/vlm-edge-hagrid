#!/usr/bin/env python3
"""Replace machine-specific absolute image paths in result CSVs.

By default, paths below the repository root become repository-relative. For
legacy rows outside the root, the tool keeps the suffix beginning with
``data/`` when one exists. ``--check`` performs no writes and exits nonzero if
an absolute image path remains.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("results/image_results")],
        help="CSV files or directories containing CSV files.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def iter_csv_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.glob("*.csv"))
        elif path.suffix.lower() == ".csv":
            files.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(files)


def sanitize_path(raw: str, root: Path) -> str:
    if not raw:
        return raw

    normalized = raw.replace("\\", "/")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        parts = candidate.parts
        if "data" in parts:
            index = parts.index("data")
            return Path(*parts[index:]).as_posix()
        return candidate.as_posix()


def process_file(path: Path, root: Path, check: bool, backup: bool) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "image_path" not in fieldnames:
            raise ValueError(f"{path} has no image_path column")
        rows = list(reader)

    changed = 0
    absolute_remaining = 0
    for row in rows:
        old = row["image_path"]
        new = sanitize_path(old, root)
        if new != old:
            changed += 1
        row["image_path"] = new
        if Path(new).is_absolute():
            absolute_remaining += 1

    if check:
        status = "OK" if absolute_remaining == 0 else "FAIL"
        print(f"{status}: {path} ({absolute_remaining} absolute paths)")
        return absolute_remaining

    if changed:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    print(f"Updated {path}: {changed} paths changed")
    return absolute_remaining


def main() -> int:
    args = parse_args()
    files = iter_csv_files(args.paths)
    if not files:
        print("No CSV files found", file=sys.stderr)
        return 1

    remaining = sum(
        process_file(path, args.root, args.check, args.backup)
        for path in files
    )
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
