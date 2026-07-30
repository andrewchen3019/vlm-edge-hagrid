#!/usr/bin/env python3

"""
Detect the largest hand in each HaGRID image, crop around it with margin,
square-pad the crop, and resize it to the requested output size.

Recommended:
    input:  336 px dataset
    output: 150 px hand crops
    margin: 0.35
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps


def resolve_source_path(
    row: dict,
    input_root: Path,
) -> Path:
    """Resolve image paths from different metadata formats."""

    stored_path = Path(row["image_path"])

    candidates = [
        stored_path,
        input_root / stored_path,
        input_root / "images" / row["label"] / stored_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find image for metadata row:\n"
        f"  stored path: {stored_path}\n"
        f"  label: {row.get('label')}"
    )


def detect_largest_hand_bbox(
    image: Image.Image,
    detector,
) -> Optional[tuple[float, float, float, float]]:
    """
    Return the largest detected hand box as pixel coordinates:
        left, top, right, bottom
    """

    rgb_array = np.asarray(image)

    result = detector.process(rgb_array)

    if not result.multi_hand_landmarks:
        return None

    width, height = image.size
    detected_boxes = []

    for hand_landmarks in result.multi_hand_landmarks:
        x_values = [
            landmark.x * width
            for landmark in hand_landmarks.landmark
        ]
        y_values = [
            landmark.y * height
            for landmark in hand_landmarks.landmark
        ]

        left = min(x_values)
        top = min(y_values)
        right = max(x_values)
        bottom = max(y_values)

        box_area = max(0.0, right - left) * max(0.0, bottom - top)

        detected_boxes.append(
            (
                box_area,
                left,
                top,
                right,
                bottom,
            )
        )

    detected_boxes.sort(reverse=True)

    _, left, top, right, bottom = detected_boxes[0]

    return left, top, right, bottom


def crop_square_with_margin(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    margin: float,
    padding_value: int,
) -> tuple[Image.Image, list[int]]:
    """
    Convert the hand box into a larger square box.

    margin=0.35 adds 35% of the hand-box size on each side.
    Padding is added if the square extends outside the image.
    """

    left, top, right, bottom = bbox

    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0

    hand_side = max(box_width, box_height)

    # Margin is applied to both sides.
    crop_side = hand_side * (1.0 + 2.0 * margin)
    crop_side = max(2, int(round(crop_side)))

    crop_left = int(round(center_x - crop_side / 2))
    crop_top = int(round(center_y - crop_side / 2))
    crop_right = crop_left + crop_side
    crop_bottom = crop_top + crop_side

    image_width, image_height = image.size

    source_left = max(0, crop_left)
    source_top = max(0, crop_top)
    source_right = min(image_width, crop_right)
    source_bottom = min(image_height, crop_bottom)

    source_crop = image.crop(
        (
            source_left,
            source_top,
            source_right,
            source_bottom,
        )
    )

    square = Image.new(
        "RGB",
        (crop_side, crop_side),
        (
            padding_value,
            padding_value,
            padding_value,
        ),
    )

    paste_x = source_left - crop_left
    paste_y = source_top - crop_top

    square.paste(
        source_crop,
        (
            paste_x,
            paste_y,
        ),
    )

    source_crop.close()

    return square, [
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
    ]


def square_pad_full_image(
    image: Image.Image,
    padding_value: int,
) -> Image.Image:
    """Fallback when no hand is detected."""

    width, height = image.size
    side = max(width, height)

    square = Image.new(
        "RGB",
        (side, side),
        (
            padding_value,
            padding_value,
            padding_value,
        ),
    )

    offset_x = (side - width) // 2
    offset_y = (side - height) // 2

    square.paste(
        image,
        (
            offset_x,
            offset_y,
        ),
    )

    return square


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/hagrid_380_resize336",
        help="Dataset containing the 336 px images",
    )

    parser.add_argument(
        "--output",
        default="data/hagrid_380_handcrop150",
        help="Output dataset directory",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=150,
        help="Final square image size",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="Extra space added on every side of the detected hand",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum MediaPipe hand-detection confidence",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--padding-value",
        type=int,
        default=127,
        help="Gray value used when square padding is needed",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    input_metadata = input_root / "metadata.jsonl"
    output_metadata = output_root / "metadata.jsonl"

    if not input_metadata.exists():
        raise FileNotFoundError(
            f"Metadata not found: {input_metadata}"
        )

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"{output_root} already exists. "
                "Use --overwrite to replace it."
            )

    output_root.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands

    detected_count = 0
    fallback_count = 0
    total_count = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.confidence,
    ) as detector, \
         input_metadata.open("r", encoding="utf-8") as input_file, \
         output_metadata.open("w", encoding="utf-8") as output_file:

        for line in input_file:
            if not line.strip():
                continue

            row = json.loads(line)

            source_path = resolve_source_path(
                row=row,
                input_root=input_root,
            )

            label = row["label"]

            output_path = (
                output_root
                / "images"
                / label
                / source_path.name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with Image.open(source_path) as opened_image:
                image = ImageOps.exif_transpose(
                    opened_image
                ).convert("RGB")

            bbox = detect_largest_hand_bbox(
                image=image,
                detector=detector,
            )

            if bbox is not None:
                prepared_image, crop_box = crop_square_with_margin(
                    image=image,
                    bbox=bbox,
                    margin=args.margin,
                    padding_value=args.padding_value,
                )

                crop_detected = True
                detected_count += 1

            else:
                prepared_image = square_pad_full_image(
                    image=image,
                    padding_value=args.padding_value,
                )

                crop_box = None
                crop_detected = False
                fallback_count += 1

            prepared_image = prepared_image.resize(
                (args.size, args.size),
                Image.Resampling.LANCZOS,
            )

            temporary_path = output_path.with_suffix(
                output_path.suffix + ".tmp"
            )

            prepared_image.save(
                temporary_path,
                format="JPEG",
                quality=args.quality,
                optimize=False,
            )

            temporary_path.replace(output_path)

            image.close()
            prepared_image.close()

            row["image_path"] = output_path.as_posix()
            row["full_path"] = output_path.as_posix()
            row["crop_path"] = output_path.as_posix()

            row["resized_width"] = args.size
            row["resized_height"] = args.size
            row["resize_max_side"] = args.size

            row["hand_crop_detected"] = crop_detected
            row["hand_crop_box"] = crop_box
            row["hand_crop_margin"] = args.margin
            row["hand_detector"] = "mediapipe_hands"

            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_count += 1

            if total_count % 20 == 0:
                print(
                    f"Processed {total_count}: "
                    f"detected={detected_count}, "
                    f"fallback={fallback_count}"
                )

    detection_rate = (
        100.0 * detected_count / total_count
        if total_count
        else 0.0
    )

    print()
    print("Finished")
    print("-----------------------------")
    print(f"Total images:       {total_count}")
    print(f"Hands detected:     {detected_count}")
    print(f"Fallback images:    {fallback_count}")
    print(f"Detection rate:     {detection_rate:.1f}%")
    print(f"Crop margin:        {args.margin}")
    print(f"Output size:        {args.size}x{args.size}")
    print(f"Output metadata:    {output_metadata}")


if __name__ == "__main__":
    main()