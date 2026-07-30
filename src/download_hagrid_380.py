#!/usr/bin/env python3

"""
Low-memory, resumable HaGRID downloader.

Downloads:
    19 classes × 20 images = 380 images

Important behavior:
- Uses hf_xet, but disables high-performance mode.
- Limits Xet download concurrency.
- Does not automatically decode every streamed image.
- Resumes by counting images already saved.
- Does not delete the existing 270 images.
- Writes metadata.jsonl after all 380 images exist.
"""

import os

# Set these before importing Hugging Face libraries.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "2"
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"

# Prevent other libraries from creating unnecessary worker threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import pyarrow as pa
except ImportError:
    pa = None


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

# Correct ClassLabel order used by the Hugging Face dataset.
FALLBACK_LABEL_NAMES = [
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


def release_memory() -> None:
    """Ask Python and PyArrow to return unused memory."""

    gc.collect()

    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass


def get_rss_mb() -> float | None:
    """Read this process's resident memory from /proc."""

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except Exception:
        pass

    return None


def clean_label(label: Any) -> str | None:
    if label is None:
        return None

    if isinstance(label, (list, tuple)):
        if not label:
            return None
        label = label[0]

    label = str(label).strip()

    for prefix in ("train_val_", "test_"):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)

    return label


def resolve_label(
    raw_label: Any,
    label_names: list[str],
) -> str | None:
    if isinstance(raw_label, int):
        if not 0 <= raw_label < len(label_names):
            return None

        return clean_label(label_names[raw_label])

    return clean_label(raw_label)


def decode_image(value: Any) -> Image.Image:
    """
    Decode only an image that is actually going to be considered for saving.
    """

    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        source_path = value.get("path")

        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as source:
                return ImageOps.exif_transpose(source).convert("RGB")

        if source_path:
            with Image.open(source_path) as source:
                return ImageOps.exif_transpose(source).convert("RGB")

    if isinstance(value, (str, Path)):
        with Image.open(value) as source:
            return ImageOps.exif_transpose(source).convert("RGB")

    raise TypeError(
        f"Unsupported image representation: {type(value).__name__}"
    )


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side > 0:
        image.thumbnail(
            (max_side, max_side),
            Image.Resampling.LANCZOS,
        )

    return image


def difference_hash(image: Image.Image) -> int:
    """
    Small perceptual fingerprint used to avoid selecting an image already
    saved by an earlier interrupted run.
    """

    small = image.convert("L").resize(
        (9, 8),
        Image.Resampling.BILINEAR,
    )

    pixels = list(small.getdata())
    small.close()

    result = 0

    for y in range(8):
        row_start = y * 9

        for x in range(8):
            left = pixels[row_start + x]
            right = pixels[row_start + x + 1]

            result <<= 1

            if left > right:
                result |= 1

    return result


def is_duplicate(
    fingerprint: int,
    existing_fingerprints: set[int],
    maximum_distance: int = 2,
) -> bool:
    for existing in existing_fingerprints:
        distance = (fingerprint ^ existing).bit_count()

        if distance <= maximum_distance:
            return True

    return False


def verify_existing_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()

        return True
    except Exception:
        return False


def scan_existing_images(
    image_root: Path,
    per_class: int,
) -> tuple[dict[str, int], dict[str, set[int]]]:
    """
    Count valid images from an earlier run and build fingerprints.

    This is what allows the script to resume from the existing 270 images.
    """

    counts: dict[str, int] = defaultdict(int)
    fingerprints: dict[str, set[int]] = defaultdict(set)

    for temporary_file in image_root.rglob("*.tmp"):
        temporary_file.unlink(missing_ok=True)

    print("Checking images from the previous run...")

    for label in LABELS:
        label_dir = image_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        valid_files = []

        for path in files:
            if not verify_existing_image(path):
                print(f"Deleting incomplete image: {path}")
                path.unlink(missing_ok=True)
                continue

            valid_files.append(path)

        if len(valid_files) > per_class:
            raise RuntimeError(
                f"{label} already contains {len(valid_files)} images, "
                f"but --per-class is {per_class}."
            )

        for path in valid_files:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                fingerprints[label].add(difference_hash(rgb))
                rgb.close()

        counts[label] = len(valid_files)

        print(f"  {label:20s}: {counts[label]}/{per_class}")

    return counts, fingerprints


def next_output_path(
    label_dir: Path,
    label: str,
) -> Path:
    index = 0

    while True:
        candidate = label_dir / f"{label}_{index:05d}.jpg"

        if not candidate.exists():
            return candidate

        index += 1


def save_image_atomic(
    image: Image.Image,
    output_path: Path,
    quality: int,
) -> None:
    """
    Save to a temporary file and rename it after the JPEG is complete.

    If the process is killed during saving, the final filename is not left
    as a corrupt partial image.
    """

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    image.save(
        temporary_path,
        format="JPEG",
        quality=quality,
        # optimize=True can cause additional CPU usage and memory spikes.
        optimize=False,
    )

    os.replace(temporary_path, output_path)


def all_complete(
    counts: dict[str, int],
    per_class: int,
) -> bool:
    return all(
        counts[label] >= per_class
        for label in LABELS
    )


def write_metadata(
    output_root: Path,
    per_class: int,
    dataset_name: str,
    split: str,
    max_side: int,
) -> Path:
    metadata_path = output_root / "metadata.jsonl"
    temporary_path = output_root / "metadata.jsonl.tmp"
    image_root = output_root / "images"

    total = 0

    with temporary_path.open("w", encoding="utf-8") as file:
        for label in LABELS:
            paths = sorted(
                path
                for path in (image_root / label).iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

            if len(paths) != per_class:
                raise RuntimeError(
                    f"{label} has {len(paths)} images; expected {per_class}."
                )

            for class_index, path in enumerate(paths):
                image_id = f"{label}_{class_index:05d}"

                row = {
                    "id": image_id,
                    "label": label,
                    "image_path": path.as_posix(),
                    "full_path": path.as_posix(),
                    "crop_path": path.as_posix(),
                    "bbox": None,
                    "class_index": class_index,
                    "source_dataset": dataset_name,
                    "source_split": split,
                    "resize_max_side": max_side,
                }

                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

                total += 1

        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, metadata_path)

    old_partial = output_root / "metadata.jsonl.partial"
    old_partial.unlink(missing_ok=True)

    print(f"Metadata rows written: {total}")

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="cj-mills/hagrid-classification-512p-no-gesture-150k",
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--out",
        default="data/hagrid_balanced20_resize336",
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
        "--shuffle-buffer",
        type=int,
        default=32,
        help=(
            "Small encoded-image shuffle buffer. "
            "Use 0 to disable shuffling."
        ),
    )

    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing images and restart from zero",
    )

    args = parser.parse_args()

    if args.per_class <= 0:
        raise ValueError("--per-class must be greater than zero")

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    output_root = Path(args.out)
    image_root = output_root / "images"

    if args.overwrite and output_root.exists():
        print(f"Removing: {output_root}")
        shutil.rmtree(output_root)

    image_root.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        (image_root / label).mkdir(parents=True, exist_ok=True)

    counts, fingerprints = scan_existing_images(
        image_root=image_root,
        per_class=args.per_class,
    )

    existing_total = sum(counts.values())
    target_total = args.per_class * len(LABELS)

    print()
    print(f"Existing valid images: {existing_total}")
    print(f"Images still needed:   {target_total - existing_total}")
    print()

    if all_complete(counts, args.per_class):
        metadata_path = write_metadata(
            output_root=output_root,
            per_class=args.per_class,
            dataset_name=args.dataset,
            split=args.split,
            max_side=args.max_side,
        )

        print("Dataset was already complete.")
        print(f"Metadata: {metadata_path}")
        return

    load_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }

    if args.token:
        load_kwargs["token"] = args.token

    print(f"Opening dataset: {args.dataset}")

    dataset = load_dataset(**load_kwargs)

    label_names = FALLBACK_LABEL_NAMES

    try:
        feature_names = dataset.features["label"].names

        if feature_names:
            label_names = list(feature_names)
    except Exception:
        pass

    # Critical memory fix:
    # Do not decode every streamed image into a PIL object.
    try:
        dataset = dataset.decode(False)
    except (AttributeError, TypeError):
        dataset = dataset.cast_column(
            "image",
            HFImage(decode=False),
        )

    # A small shuffle buffer is safe because it contains encoded images,
    # not decoded PIL images.
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
        )

    scanned = 0
    newly_saved = 0

    progress = tqdm(
        total=target_total,
        initial=existing_total,
        desc="HaGRID images",
        unit="image",
    )

    try:
        for example in dataset:
            scanned += 1

            raw_label = example.get(
                "label",
                example.get("labels"),
            )

            label = resolve_label(
                raw_label,
                label_names,
            )

            if label not in LABELS:
                continue

            # Do not decode the image when this class is already complete.
            if counts[label] >= args.per_class:
                continue

            image_value = example.get("image")

            if image_value is None:
                continue

            image = None

            try:
                image = decode_image(image_value)
                image = resize_image(image, args.max_side)

                fingerprint = difference_hash(image)

                if is_duplicate(
                    fingerprint,
                    fingerprints[label],
                ):
                    continue

                label_dir = image_root / label
                output_path = next_output_path(
                    label_dir,
                    label,
                )

                save_image_atomic(
                    image=image,
                    output_path=output_path,
                    quality=args.quality,
                )

                fingerprints[label].add(fingerprint)
                counts[label] += 1
                newly_saved += 1

                progress.update(1)
                progress.set_postfix(
                    label=label,
                    scanned=scanned,
                )

            except Exception as error:
                tqdm.write(
                    f"Skipping {label} image at stream row "
                    f"{scanned}: {error}"
                )

            finally:
                if image is not None:
                    image.close()

                del example
                del image_value

            if newly_saved % 5 == 0:
                release_memory()

            if scanned % 2000 == 0:
                release_memory()

                rss = get_rss_mb()

                if rss is not None:
                    tqdm.write(
                        f"Scanned {scanned}; process RAM: {rss:.0f} MB"
                    )

            if all_complete(counts, args.per_class):
                break

    finally:
        progress.close()
        release_memory()

    print()
    print("Current counts:")

    for label in LABELS:
        print(f"{label:20s}: {counts[label]}/{args.per_class}")

    if not all_complete(counts, args.per_class):
        raise RuntimeError(
            "The stream ended before every class reached the target.\n"
            "Rerun the same command; existing images will be preserved."
        )

    metadata_path = write_metadata(
        output_root=output_root,
        per_class=args.per_class,
        dataset_name=args.dataset,
        split=args.split,
        max_side=args.max_side,
    )

    print()
    print("Done.")
    print(f"Existing before run: {existing_total}")
    print(f"New images saved:   {newly_saved}")
    print(f"Total images:       {target_total}")
    print(f"Metadata:           {metadata_path}")


if __name__ == "__main__":
    main()