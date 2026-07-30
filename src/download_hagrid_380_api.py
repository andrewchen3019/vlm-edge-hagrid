#!/usr/bin/env python3

import argparse
import io
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps


DATASET = "cj-mills/hagrid-classification-512p-no-gesture-150k"
API_URL = "https://datasets-server.huggingface.co/filter"

# Exact ClassLabel order in the Hugging Face dataset.
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


def api_headers():
    token = os.environ.get("HF_TOKEN")

    if token:
        return {"Authorization": f"Bearer {token}"}

    return {}


def get_filtered_rows(label_index, offset, length):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "where": f'"label"={label_index}',
        "offset": offset,
        "length": length,
    }

    last_error = None

    for attempt in range(6):
        try:
            response = requests.get(
                API_URL,
                params=params,
                headers=api_headers(),
                timeout=(15, 120),
            )
            response.raise_for_status()

            data = response.json()

            if "rows" not in data:
                raise RuntimeError(
                    f"Unexpected API response: {data}"
                )

            return data

        except Exception as exc:
            last_error = exc

            if attempt < 5:
                delay = 2 ** attempt
                print(
                    f"API request failed; retrying in {delay}s: {exc}"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Dataset API request failed: {last_error}"
    )


def select_rows(per_class, seed):
    rng = random.Random(seed)
    selected = []

    for label_index, label in enumerate(LABELS):
        # First request obtains the total number of matching rows.
        probe = get_filtered_rows(
            label_index=label_index,
            offset=0,
            length=1,
        )

        total_available = int(probe["num_rows_total"])

        if total_available < per_class:
            raise RuntimeError(
                f"{label} contains only {total_available} rows; "
                f"requested {per_class}."
            )

        # Select a reproducible random section instead of always taking
        # the first images in each class.
        max_offset = total_available - per_class
        offset = rng.randint(0, max_offset)

        data = get_filtered_rows(
            label_index=label_index,
            offset=offset,
            length=per_class,
        )

        rows = data["rows"]

        if len(rows) != per_class:
            raise RuntimeError(
                f"Received {len(rows)} rows for {label}; "
                f"expected {per_class}."
            )

        for class_index, result in enumerate(rows):
            row = result["row"]
            image_info = row.get("image")

            if not isinstance(image_info, dict):
                raise RuntimeError(
                    f"Missing image information for {label}: {row}"
                )

            image_url = image_info.get("src")

            if not image_url:
                raise RuntimeError(
                    f"Missing image URL for {label}: {image_info}"
                )

            selected.append(
                {
                    "label": label,
                    "label_index": label_index,
                    "class_index": class_index,
                    "source_row_index": result["row_idx"],
                    "url": image_url,
                }
            )

        print(
            f"Selected {per_class:2d} {label:20s} images "
            f"from offset {offset}/{total_available}"
        )

    return selected


def download_image(item, output_root, max_side, quality):
    label = item["label"]
    source_row_index = item["source_row_index"]

    image_id = f"{label}_{source_row_index:06d}"
    output_path = (
        output_root
        / "images"
        / label
        / f"{image_id}.jpg"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(6):
        try:
            response = requests.get(
                item["url"],
                timeout=(15, 120),
            )
            response.raise_for_status()

            with Image.open(io.BytesIO(response.content)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

                if max_side > 0:
                    image.thumbnail(
                        (max_side, max_side),
                        Image.Resampling.LANCZOS,
                    )

                width, height = image.size

                # Saving without optimize uses less CPU and memory.
                image.save(
                    output_path,
                    format="JPEG",
                    quality=quality,
                    optimize=False,
                )

                image.close()

            return {
                "id": image_id,
                "label": label,
                "label_index": item["label_index"],
                "class_index": item["class_index"],
                "image_path": output_path.as_posix(),
                "full_path": output_path.as_posix(),
                "crop_path": output_path.as_posix(),
                "bbox": None,
                "resized_width": width,
                "resized_height": height,
                "resize_max_side": max_side,
                "source_dataset": DATASET,
                "source_split": "train",
                "source_row_index": source_row_index,
            }

        except Exception as exc:
            last_error = exc
            output_path.unlink(missing_ok=True)

            if attempt < 5:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to download {label} row "
        f"{source_row_index}: {last_error}"
    )


def main():
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
        "--workers",
        type=int,
        default=4,
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

    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    output_root = Path(args.out)

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to replace it."
            )

    output_root.mkdir(parents=True, exist_ok=True)

    total_target = len(LABELS) * args.per_class

    print(f"Dataset:          {DATASET}")
    print(f"Classes:          {len(LABELS)}")
    print(f"Images per class: {args.per_class}")
    print(f"Total target:     {total_target}")
    print()

    selected = select_rows(
        per_class=args.per_class,
        seed=args.seed,
    )

    print()
    print(f"Downloading {len(selected)} images...")

    metadata_rows = []
    completed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(
                download_image,
                item,
                output_root,
                args.max_side,
                args.quality,
            )
            for item in selected
        ]

        for future in as_completed(futures):
            row = future.result()
            metadata_rows.append(row)
            completed += 1

            if completed % 20 == 0 or completed == total_target:
                print(
                    f"Downloaded {completed}/{total_target}"
                )

    metadata_rows.sort(
        key=lambda row: (
            row["label_index"],
            row["class_index"],
        )
    )

    metadata_path = output_root / "metadata.jsonl"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        for row in metadata_rows:
            metadata_file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    if len(metadata_rows) != total_target:
        raise RuntimeError(
            f"Created {len(metadata_rows)} rows; "
            f"expected {total_target}."
        )

    print()
    print("Done.")
    print(f"Images downloaded: {len(metadata_rows)}")
    print(f"Images:            {output_root / 'images'}")
    print(f"Metadata:          {metadata_path}")


if __name__ == "__main__":
    main()
