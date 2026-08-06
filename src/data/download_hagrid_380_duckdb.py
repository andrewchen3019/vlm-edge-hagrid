#!/usr/bin/env python3
"""Create the deterministic 380-image HaGRID benchmark subset.

The script reads only the required rows from the remote Parquet dataset,
orders each class by the upstream image path, resizes images to a maximum side
of 336 pixels, and records source/output SHA-256 hashes in metadata.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
from PIL import Image, ImageOps

DATASET_ID = "cj-mills/hagrid-classification-512p-no-gesture-150k"
DEFAULT_DATASET_GLOB = f"hf://datasets/{DATASET_ID}/data/*.parquet"

# Exact label-index mapping published by the source dataset.
LABELS: tuple[str, ...] = (
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "no_gesture",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
)


class DatasetPreparationError(RuntimeError):
    """Raised when the benchmark subset cannot be generated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/hagrid_380_resize336"),
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=336)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help="DuckDB hf:// Parquet glob for the source dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before rebuilding it.",
    )
    args = parser.parse_args()

    if args.per_class <= 0:
        parser.error("--per-class must be positive")
    if args.max_side <= 0:
        parser.error("--max-side must be positive")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    return args


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def open_duckdb() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    try:
        connection.execute("LOAD httpfs")
    except duckdb.Error:
        print("Installing DuckDB httpfs extension...")
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")

    connection.execute("SET threads = 2")
    connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET enable_object_cache = true")
    return connection


def decode_image(image_bytes: Any) -> Image.Image:
    if image_bytes is None:
        raise DatasetPreparationError("Source image bytes are missing")
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    if not isinstance(image_bytes, bytes):
        image_bytes = bytes(image_bytes)

    with Image.open(io.BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def fetch_class_rows(
    connection: duckdb.DuckDBPyConnection,
    dataset_glob: str,
    label_index: int,
    per_class: int,
) -> list[tuple[bytes, str]]:
    # Ordering by the upstream path makes class selection repeatable. The old
    # LIMIT/OFFSET-only query could return a different subset after an upstream
    # Parquet rewrite or query-planner change.
    query = """
        SELECT image_bytes, source_path
        FROM (
            SELECT
                struct_extract(image, 'bytes') AS image_bytes,
                struct_extract(image, 'path') AS source_path
            FROM read_parquet(?)
            WHERE label = ?
        )
        ORDER BY source_path
        LIMIT ?
    """
    rows = connection.execute(
        query,
        [dataset_glob, label_index, per_class],
    ).fetchall()

    if len(rows) != per_class:
        raise DatasetPreparationError(
            f"Label index {label_index} returned {len(rows)} rows; "
            f"expected {per_class}"
        )
    return [(bytes(image_bytes), str(source_path)) for image_bytes, source_path in rows]


def save_jpeg(
    image_bytes: bytes,
    output_path: Path,
    max_side: int,
    quality: int,
) -> None:
    image = decode_image(image_bytes)
    try:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        image.save(
            temporary_path,
            format="JPEG",
            quality=quality,
            optimize=False,
        )
        temporary_path.replace(output_path)
    finally:
        image.close()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = args.out.expanduser()
    if args.overwrite and output_root.exists():
        print(f"Removing old output: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    connection = open_duckdb()
    try:
        for label_index, label in enumerate(LABELS):
            print(f"{label:20s}: selecting {args.per_class} deterministic rows")
            rows = fetch_class_rows(
                connection,
                args.dataset_glob,
                label_index,
                args.per_class,
            )

            for class_index, (image_bytes, source_path) in enumerate(rows):
                image_id = f"{label}_{class_index:05d}"
                output_path = (
                    output_root
                    / "images"
                    / label
                    / f"{image_id}.jpg"
                )
                save_jpeg(
                    image_bytes,
                    output_path,
                    args.max_side,
                    args.quality,
                )

                metadata_rows.append(
                    {
                        "id": image_id,
                        "label": label,
                        "label_index": label_index,
                        "class_index": class_index,
                        "image_path": portable_path(output_path),
                        "resize_max_side": args.max_side,
                        "jpeg_quality": args.quality,
                        "source_dataset": DATASET_ID,
                        "source_split": "train",
                        "source_path": source_path,
                        "source_sha256": sha256_bytes(image_bytes),
                        "output_sha256": sha256_file(output_path),
                    }
                )
    finally:
        connection.close()

    expected = len(LABELS) * args.per_class
    if len(metadata_rows) != expected:
        raise DatasetPreparationError(
            f"Created {len(metadata_rows)} rows; expected {expected}"
        )

    metadata_path = output_root / "metadata.jsonl"
    temporary_metadata = output_root / "metadata.jsonl.tmp"
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_metadata.replace(metadata_path)

    manifest_path = output_root / "dataset_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "source_dataset": DATASET_ID,
            "dataset_glob": args.dataset_glob,
            "classes": list(LABELS),
            "num_classes": len(LABELS),
            "images_per_class": args.per_class,
            "total_images": expected,
            "max_side": args.max_side,
            "jpeg_quality": args.quality,
            "selection_rule": "ORDER BY upstream image path, first N per class",
            "metadata_sha256": sha256_file(metadata_path),
        },
    )
    return metadata_path, manifest_path


def main() -> int:
    args = parse_args()
    metadata_path, manifest_path = prepare_dataset(args)
    print()
    print("Done")
    print(f"Classes:          {len(LABELS)}")
    print(f"Images per class: {args.per_class}")
    print(f"Total images:     {len(LABELS) * args.per_class}")
    print(f"Metadata:         {metadata_path}")
    print(f"Manifest:         {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
