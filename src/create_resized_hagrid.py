#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image


def resolve_path(path_str, metadata_dir):
    p = Path(path_str)

    if p.is_absolute() and p.exists():
        return p

    # Usually your paths are relative to project root.
    if p.exists():
        return p

    # Fallback: relative to metadata file directory.
    p2 = metadata_dir / p
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Could not find image: {path_str}")


def resize_image(src_path, dst_path, max_side, quality, overwrite=False):
    if dst_path.exists() and not overwrite:
        with Image.open(dst_path) as im:
            return {
                "resized_width": im.width,
                "resized_height": im.height,
                "skipped_existing": True,
            }

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        orig_w, orig_h = im.size

        # Convert to RGB for JPEG compatibility.
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")

        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        resized_w, resized_h = im.size

        im.save(
            dst_path,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

    return {
        "orig_width": orig_w,
        "orig_height": orig_h,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "skipped_existing": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Input metadata.jsonl")
    parser.add_argument("--out-root", required=True, help="Output dataset root")
    parser.add_argument("--max-side", type=int, required=True, help="Max width/height after resizing")
    parser.add_argument("--input-field", default="image_path", help="Metadata field containing image path")
    parser.add_argument("--output-field", default="image_path", help="Metadata field to write resized path into")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    metadata_dir = metadata_path.parent
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_metadata = out_root / "metadata.jsonl"

    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.limit is not None:
        rows = rows[: args.limit]

    written = 0
    skipped = 0

    with out_metadata.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            if args.input_field not in row:
                raise KeyError(f"Row missing field {args.input_field}: {row}")

            src_str = row[args.input_field]
            src_path = resolve_path(src_str, metadata_dir)

            label = row.get("label", src_path.parent.name)
            dst_path = out_images / label / src_path.name

            info = resize_image(
                src_path=src_path,
                dst_path=dst_path,
                max_side=args.max_side,
                quality=args.quality,
                overwrite=args.overwrite,
            )

            new_row = dict(row)
            new_row["original_image_path"] = src_str
            new_row[args.output_field] = dst_path.as_posix()
            new_row["resize_max_side"] = args.max_side

            # Add dimensions if available.
            if "orig_width" in info:
                new_row["orig_width"] = info["orig_width"]
                new_row["orig_height"] = info["orig_height"]

            new_row["resized_width"] = info["resized_width"]
            new_row["resized_height"] = info["resized_height"]

            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")

            if info["skipped_existing"]:
                skipped += 1
            else:
                written += 1

            if i % 500 == 0:
                print(f"Processed {i}/{len(rows)} images...")

    print()
    print("Done.")
    print(f"Input metadata:  {metadata_path}")
    print(f"Output metadata: {out_metadata}")
    print(f"Output images:   {out_images}")
    print(f"Max side:        {args.max_side}")
    print(f"Images written:  {written}")
    print(f"Images skipped:  {skipped}")
    print(f"Total rows:      {len(rows)}")


if __name__ == "__main__":
    main()
