#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def resize_image(src: Path, dst: Path, max_side: int, quality: int):
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=quality, optimize=False)

        return img.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/hagrid_380_resize336",
        help="Existing dataset directory",
    )
    parser.add_argument(
        "--output",
        default="data/hagrid_380_resize225",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=225,
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

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(input_root)

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise RuntimeError(
                f"{output_root} already exists. "
                "Use --overwrite to replace it."
            )

    output_root.mkdir(parents=True)

    input_metadata = input_root / "metadata.jsonl"
    output_metadata = output_root / "metadata.jsonl"

    count = 0

    with open(input_metadata, "r", encoding="utf-8") as fin, \
         open(output_metadata, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            old_path = Path(row["image_path"])

            # Preserve relative directory structure
            relative = old_path.relative_to(input_root)
            new_path = output_root / relative

            width, height = resize_image(
                old_path,
                new_path,
                args.max_side,
                args.quality,
            )

            row["image_path"] = str(new_path)
            row["full_path"] = str(new_path)
            row["crop_path"] = str(new_path)
            row["resized_width"] = width
            row["resized_height"] = height
            row["resize_max_side"] = args.max_side

            fout.write(json.dumps(row) + "\n")
            count += 1

    print(f"Done.")
    print(f"Images resized: {count}")
    print(f"Output dataset: {output_root}")
    print(f"Metadata: {output_metadata}")


if __name__ == "__main__":
    main()