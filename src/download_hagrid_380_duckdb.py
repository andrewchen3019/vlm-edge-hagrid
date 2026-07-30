#!/usr/bin/env python3

import argparse
import io
import json
import shutil
from pathlib import Path

import duckdb
from PIL import Image, ImageOps


DATASET_GLOB = (
    "hf://datasets/"
    "cj-mills/hagrid-classification-512p-no-gesture-150k/"
    "data/*.parquet"
)

# Exact label-index mapping from the dataset.
LABELS = [
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
]


def open_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # Install only when it is not already available.
    try:
        con.execute("LOAD httpfs")
    except duckdb.Error:
        print("Installing DuckDB httpfs extension...")
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")

    # Keep resource use safe for the Jetson.
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '1GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_object_cache = true")

    return con


def decode_image(image_bytes) -> Image.Image:
    if image_bytes is None:
        raise ValueError("Image bytes are missing")

    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()

    with Image.open(io.BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def existing_count(label_directory: Path) -> int:
    return len(
        [
            path
            for path in label_directory.glob("*.jpg")
            if path.is_file()
        ]
    )


def download_class(
    con: duckdb.DuckDBPyConnection,
    label_index: int,
    label: str,
    output_root: Path,
    per_class: int,
    max_side: int,
    quality: int,
) -> int:
    label_directory = output_root / "images" / label
    label_directory.mkdir(parents=True, exist_ok=True)

    already_saved = existing_count(label_directory)

    if already_saved >= per_class:
        print(f"{label:20s}: already complete")
        return already_saved

    needed = per_class - already_saved

    print(
        f"{label:20s}: downloading {needed} "
        f"(already have {already_saved})"
    )

    query = """
        SELECT
            struct_extract(image, 'bytes') AS image_bytes,
            struct_extract(image, 'path') AS source_path
        FROM read_parquet(?)
        WHERE label = ?
        LIMIT ?
        OFFSET ?
    """

    rows = con.execute(
        query,
        [
            DATASET_GLOB,
            label_index,
            needed,
            already_saved,
        ],
    ).fetchall()

    if len(rows) < needed:
        raise RuntimeError(
            f"Only received {len(rows)} images for {label}; "
            f"needed {needed}."
        )

    saved = already_saved

    for image_bytes, source_path in rows:
        image = decode_image(image_bytes)

        try:
            if max_side > 0:
                image.thumbnail(
                    (max_side, max_side),
                    Image.Resampling.LANCZOS,
                )

            output_path = (
                label_directory
                / f"{label}_{saved:05d}.jpg"
            )

            temporary_path = output_path.with_suffix(".jpg.tmp")

            image.save(
                temporary_path,
                format="JPEG",
                quality=quality,
                optimize=False,
            )

            temporary_path.replace(output_path)

            saved += 1

        finally:
            image.close()

    return saved


def write_metadata(
    output_root: Path,
    per_class: int,
    max_side: int,
) -> Path:
    metadata_path = output_root / "metadata.jsonl"
    temporary_path = output_root / "metadata.jsonl.tmp"

    total = 0

    with temporary_path.open("w", encoding="utf-8") as file:
        for label_index, label in enumerate(LABELS):
            label_directory = output_root / "images" / label
            paths = sorted(label_directory.glob("*.jpg"))

            if len(paths) != per_class:
                raise RuntimeError(
                    f"{label} has {len(paths)} images; "
                    f"expected {per_class}."
                )

            for class_index, path in enumerate(paths):
                image_id = f"{label}_{class_index:05d}"

                row = {
                    "id": image_id,
                    "label": label,
                    "label_index": label_index,
                    "class_index": class_index,
                    "image_path": path.as_posix(),
                    "full_path": path.as_posix(),
                    "crop_path": path.as_posix(),
                    "bbox": None,
                    "resize_max_side": max_side,
                    "source_dataset": (
                        "cj-mills/"
                        "hagrid-classification-512p-"
                        "no-gesture-150k"
                    ),
                    "source_split": "train",
                }

                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

                total += 1

    temporary_path.replace(metadata_path)

    expected = len(LABELS) * per_class

    if total != expected:
        raise RuntimeError(
            f"Metadata contains {total} rows; expected {expected}."
        )

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="data/hagrid_380_resize336",
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=336,
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=92,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    output_root = Path(args.out)

    if args.overwrite and output_root.exists():
        print(f"Removing old output: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    print("Opening remote HaGRID Parquet files...")
    print("Only the required Parquet ranges will be read.")
    print()

    con = open_duckdb()

    try:
        counts = {}

        for label_index, label in enumerate(LABELS):
            counts[label] = download_class(
                con=con,
                label_index=label_index,
                label=label,
                output_root=output_root,
                per_class=args.per_class,
                max_side=args.max_side,
                quality=args.quality,
            )

        metadata_path = write_metadata(
            output_root=output_root,
            per_class=args.per_class,
            max_side=args.max_side,
        )

    finally:
        con.close()

    print()
    print("Done.")
    print("----------------------------")

    for label in LABELS:
        print(f"{label:20s}: {counts[label]}")

    total = sum(counts.values())

    print()
    print(f"Classes:          {len(LABELS)}")
    print(f"Images per class: {args.per_class}")
    print(f"Total images:     {total}")
    print(f"Metadata:         {metadata_path}")


if __name__ == "__main__":
    main()
