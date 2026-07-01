import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_label(label):
    """
    Handles labels like:
      call
      train_val_call
      0
      ['call']
    """
    if isinstance(label, list):
        if len(label) == 0:
            return None
        label = label[0]

    if label is None:
        return None

    label = str(label)

    if label.startswith("train_val_"):
        label = label.replace("train_val_", "")

    if label.startswith("test_"):
        label = label.replace("test_", "")

    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cj-mills/hagrid-classification-512p-no-gesture-150k-zip",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/hagrid_day1")
    parser.add_argument("--max-per-class", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out)
    image_dir = out / "images"
    ensure_dir(image_dir)

    for label in LABELS:
        ensure_dir(image_dir / label)

    print(f"Loading dataset: {args.dataset}")
    print("Using streaming=True so it does not download the whole dataset.")

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Try to read ClassLabel names if HF gives labels as integers.
    label_names = None
    try:
        feature = ds.features["label"]
        if hasattr(feature, "names"):
            label_names = feature.names
            print("Detected label names:", label_names)
    except Exception:
        pass

    counts = defaultdict(int)
    total_target = args.max_per_class * len(LABELS)
    metadata_path = out / "metadata.jsonl"

    with open(metadata_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(ds, total=total_target)):
            raw_label = ex.get("label", ex.get("labels", None))

            if isinstance(raw_label, int) and label_names is not None:
                label = clean_label(label_names[raw_label])
            else:
                label = clean_label(raw_label)

            if label not in LABELS:
                continue

            if counts[label] >= args.max_per_class:
                continue

            img = ex.get("image", None)
            if img is None:
                continue

            if not isinstance(img, Image.Image):
                try:
                    img = Image.open(img)
                except Exception:
                    continue

            img = img.convert("RGB")

            image_id = ex.get("id", f"{label}_{counts[label]:05d}")
            safe_id = str(image_id).replace("/", "_")

            out_path = image_dir / label / f"{safe_id}.jpg"
            img.save(out_path, quality=90)

            row = {
                "id": safe_id,
                "label": label,
                "image_path": str(out_path),
                "full_path": str(out_path),
                "crop_path": str(out_path),
                "bbox": None,
            }

            f.write(json.dumps(row) + "\n")
            counts[label] += 1

            if sum(counts.values()) >= total_target:
                break

    print("\nDone.")
    print("Counts:")
    for label in LABELS:
        print(f"{label}: {counts[label]}")

    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()