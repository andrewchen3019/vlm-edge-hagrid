#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps


LABELS = [
    "call",
    "no_gesture",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def find_images(root: Path) -> dict[str, list[Path]]:
    """Find HaGRID images stored inside class-named directories."""

    images = {label: [] for label in LABELS}
    wanted = set(LABELS)

    print(f"Scanning {root} for HaGRID class folders...")

    for directory, subdirectories, filenames in os.walk(root):
        # Skip repository and cache folders.
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in {".git", "__pycache__", ".cache"}
        ]

        directory_path = Path(directory)
        label = directory_path.name

        if label not in wanted:
            continue

        for filename in filenames:
            path = directory_path / filename

            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images[label].append(path)

        # Images should be directly inside each class directory.
        subdirectories[:] = []

    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced 380-image HaGRID dataset."
    )

    parser.add_argument(
        "--root",
        default="repos/hagrid",
        help="Root containing the full local HaGRID dataset",
    )

    parser.add_argument(
        "--out",
        default="data/hagrid_380_resize336",
        help="Output dataset directory",
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
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = Path(args.out)
    output_images = output_root / "images"
    metadata_path = output_root / "metadata.jsonl"

    if not root.exists():
        raise FileNotFoundError(f"HaGRID directory not found: {root}")

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(
                f"Output directory already exists: {output_root}\n"
                "Use --overwrite to replace it."
            )

    output_images.mkdir(parents=True, exist_ok=True)

    available = find_images(root)

    print()
    print("Images found:")

    for label in LABELS:
        count = len(available[label])
        print(f"{label:20s}: {count}")

        if count < args.per_class:
            raise RuntimeError(
                f"Only found {count} images for {label}; "
                f"need {args.per_class}.\n"
                "Check that the HaGRID image archives are extracted "
                "under repos/hagrid."
            )

    rng = random.Random(args.seed)
    metadata_rows = []

    print()
    print("Creating balanced dataset...")

    for label in LABELS:
        label_directory = output_images / label
        label_directory.mkdir(parents=True, exist_ok=True)

        candidates = sorted(available[label])
        selected = rng.sample(candidates, args.per_class)

        for index, source_path in enumerate(selected):
            image_id = f"{label}_{index:05d}"
            output_path = label_directory / f"{image_id}.jpg"

            with Image.open(source_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")

                if args.max_side > 0:
                    image.thumbnail(
                        (args.max_side, args.max_side),
                        Image.Resampling.LANCZOS,
                    )

                width, height = image.size

                image.save(
                    output_path,
                    format="JPEG",
                    quality=args.quality,
                    optimize=False,
                )

                image.close()

            metadata_rows.append(
                {
                    "id": image_id,
                    "label": label,
                    "image_path": output_path.as_posix(),
                    "full_path": output_path.as_posix(),
                    "crop_path": output_path.as_posix(),
                    "bbox": None,
                    "resized_width": width,
                    "resized_height": height,
                    "resize_max_side": args.max_side,
                    "original_image_path": source_path.as_posix(),
                }
            )

        print(f"Saved {args.per_class:2d} images: {label}")

    # Shuffle metadata order deterministically.
    rng.shuffle(metadata_rows)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for row in metadata_rows:
            metadata_file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    expected_total = len(LABELS) * args.per_class

    if len(metadata_rows) != expected_total:
        raise RuntimeError(
            f"Created {len(metadata_rows)} images; "
            f"expected {expected_total}."
        )

    print()
    print("Done.")
    print(f"Classes:          {len(LABELS)}")
    print(f"Images per class: {args.per_class}")
    print(f"Total images:     {len(metadata_rows)}")
    print(f"Images:           {output_images}")
    print(f"Metadata:         {metadata_path}")


if __name__ == "__main__":
    main()
