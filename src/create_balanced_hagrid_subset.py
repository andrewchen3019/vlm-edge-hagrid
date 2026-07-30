#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_CLASSES = [
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
    ".bmp",
}


def find_label_in_path(path: Path, labels: set[str]) -> str | None:
    """
    Find a class label among the image's parent directory names.

    For example:
        repos/hagrid/images/train/call/image.jpg
    will be labeled as:
        call
    """
    for part in reversed(path.parts[:-1]):
        if part in labels:
            return part

    return None


def collect_images(
    source_root: Path,
    labels: list[str],
) -> dict[str, list[Path]]:
    labels_set = set(labels)
    grouped: dict[str, list[Path]] = defaultdict(list)

    print(f"Scanning: {source_root}")

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = find_label_in_path(path, labels_set)

        if label is not None:
            grouped[label].append(path)

    return grouped


def unique_destination(
    output_directory: Path,
    source_path: Path,
    used_names: set[str],
) -> Path:
    """
    Prevent duplicate filenames from overwriting each other.
    """
    original_name = source_path.name
    candidate_name = original_name
    counter = 1

    while candidate_name in used_names:
        candidate_name = (
            f"{source_path.stem}_{counter:04d}{source_path.suffix.lower()}"
        )
        counter += 1

    used_names.add(candidate_name)
    return output_directory / candidate_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced HaGRID subset by selecting a fixed number "
            "of images from each class."
        )
    )

    parser.add_argument(
        "--source-root",
        required=True,
        help="Root of the full HaGRID repository or image dataset",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        help="Output directory for selected images and metadata.jsonl",
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Number of images to select per class",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random-selection seed",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class labels to include",
    )

    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying image files",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before creating the subset",
    )

    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_metadata = out_root / "metadata.jsonl"

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_root}"
        )

    if args.per_class <= 0:
        raise ValueError("--per-class must be greater than zero")

    if out_root.exists() and args.overwrite:
        print(f"Removing existing output: {out_root}")
        shutil.rmtree(out_root)

    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {out_root}\n"
            "Use --overwrite to replace it."
        )

    random_generator = random.Random(args.seed)

    grouped = collect_images(
        source_root=source_root,
        labels=args.classes,
    )

    print()
    print("Images found:")

    missing_classes: list[str] = []

    for label in args.classes:
        count = len(grouped.get(label, []))
        print(f"  {label:20s} {count}")

        if count < args.per_class:
            missing_classes.append(label)

    if missing_classes:
        descriptions = [
            f"{label}: found {len(grouped.get(label, []))}, "
            f"need {args.per_class}"
            for label in missing_classes
        ]

        raise RuntimeError(
            "\nNot enough images for these classes:\n  "
            + "\n  ".join(descriptions)
            + "\n\nCheck the repository directory structure and class names."
        )

    out_images.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []

    for label in args.classes:
        candidates = sorted(grouped[label])

        selected = random_generator.sample(
            candidates,
            args.per_class,
        )

        label_output_directory = out_images / label
        label_output_directory.mkdir(parents=True, exist_ok=True)

        used_names: set[str] = set()

        for class_index, source_path in enumerate(selected):
            destination_path = unique_destination(
                output_directory=label_output_directory,
                source_path=source_path,
                used_names=used_names,
            )

            if args.symlink:
                destination_path.symlink_to(source_path)
            else:
                shutil.copy2(source_path, destination_path)

            metadata_rows.append(
                {
                    "image_path": destination_path.as_posix(),
                    "label": label,
                    "class_index": class_index,
                    "original_image_path": source_path.as_posix(),
                }
            )

    # Randomize metadata order while keeping selection reproducible.
    random_generator.shuffle(metadata_rows)

    with out_metadata.open("w", encoding="utf-8") as output_file:
        for row in metadata_rows:
            output_file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    expected_total = len(args.classes) * args.per_class

    print()
    print("Done.")
    print(f"Classes:          {len(args.classes)}")
    print(f"Images per class: {args.per_class}")
    print(f"Total images:     {len(metadata_rows)}")
    print(f"Expected total:   {expected_total}")
    print(f"Output images:    {out_images}")
    print(f"Output metadata:  {out_metadata}")
    print(f"Random seed:      {args.seed}")
    print(
        "Storage mode:    "
        + ("symbolic links" if args.symlink else "copied files")
    )


if __name__ == "__main__":
    main()