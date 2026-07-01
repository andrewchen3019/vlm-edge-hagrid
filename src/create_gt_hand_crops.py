import argparse
import json
from pathlib import Path

from PIL import Image


BBOX_KEYS = [
    "bbox",
    "bboxes",
    "box",
    "boxes",
    "hand_bbox",
    "hand_bboxes",
]


def find_bbox(row):
    for key in BBOX_KEYS:
        if key in row and row[key] is not None:
            bbox = row[key]

            # If list of boxes, choose the largest one
            if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
                boxes = bbox
                best = None
                best_area = -1
                for b in boxes:
                    if len(b) >= 4:
                        area = abs(float(b[2]) * float(b[3]))
                        if area > best_area:
                            best = b
                            best_area = area
                return best

            # Single list box
            if isinstance(bbox, list) and len(bbox) >= 4:
                return bbox

            # Dict box
            if isinstance(bbox, dict):
                return bbox

    return None


def dict_bbox_to_list(bbox):
    # Common dict formats
    if all(k in bbox for k in ["x", "y", "w", "h"]):
        return [bbox["x"], bbox["y"], bbox["w"], bbox["h"]], "xywh"

    if all(k in bbox for k in ["x", "y", "width", "height"]):
        return [bbox["x"], bbox["y"], bbox["width"], bbox["height"]], "xywh"

    if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
        return [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], "xyxy"

    if all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
        return [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]], "xyxy"

    raise ValueError(f"Unsupported bbox dict format: {bbox}")


def convert_bbox_to_xyxy(bbox, img_w, img_h, bbox_format):
    if isinstance(bbox, dict):
        bbox, detected_format = dict_bbox_to_list(bbox)
        if bbox_format == "auto":
            bbox_format = detected_format

    vals = [float(x) for x in bbox[:4]]

    # If coordinates look normalized, scale them.
    normalized = max(vals) <= 2.0

    if bbox_format == "auto":
        # HaGRID-style boxes are often xywh.
        # If x+w and y+h look valid, assume xywh.
        x, y, a, b = vals
        if normalized:
            if x + a <= 1.5 and y + b <= 1.5:
                bbox_format = "xywh"
            else:
                bbox_format = "xyxy"
        else:
            # For pixel boxes, prefer xywh unless a,b look like bottom-right coords.
            if a > x and b > y and a <= img_w and b <= img_h:
                # Could be xyxy, but HaGRID commonly uses xywh.
                bbox_format = "xywh"
            else:
                bbox_format = "xywh"

    if normalized:
        if bbox_format == "xywh":
            x, y, w, h = vals
            x1 = x * img_w
            y1 = y * img_h
            x2 = (x + w) * img_w
            y2 = (y + h) * img_h
        elif bbox_format == "xyxy":
            x1, y1, x2, y2 = vals
            x1 *= img_w
            x2 *= img_w
            y1 *= img_h
            y2 *= img_h
        else:
            raise ValueError(f"Unknown bbox_format: {bbox_format}")
    else:
        if bbox_format == "xywh":
            x, y, w, h = vals
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif bbox_format == "xyxy":
            x1, y1, x2, y2 = vals
        else:
            raise ValueError(f"Unknown bbox_format: {bbox_format}")

    return x1, y1, x2, y2


def add_padding(x1, y1, x2, y2, img_w, img_h, pad_frac):
    bw = x2 - x1
    bh = y2 - y1

    px = bw * pad_frac
    py = bh * pad_frac

    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(img_w, int(round(x2 + px)))
    y2 = min(img_h, int(round(y2 + py)))

    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/hagrid_day1/metadata.jsonl")
    parser.add_argument("--crop-dir", default="data/hagrid_day1_gt_crops")
    parser.add_argument("--out-metadata", default="data/hagrid_day1/metadata_gt_crops.jsonl")
    parser.add_argument("--bbox-format", default="xywh", choices=["xywh", "xyxy", "auto"])
    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--fallback-full-image", action="store_true")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    crop_dir = Path(args.crop_dir)
    out_metadata = Path(args.out_metadata)

    crop_dir.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    cropped = 0
    fallback = 0
    skipped = 0

    with open(metadata_path, "r", encoding="utf-8") as fin, open(out_metadata, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)

            image_path = row.get("image_path") or row.get("crop_path") or row.get("full_path")
            label = row.get("label", "unknown")

            if image_path is None:
                skipped += 1
                continue

            image_path = Path(image_path)

            if not image_path.exists():
                skipped += 1
                continue

            bbox = find_bbox(row)

            # If no box exists, either skip or keep full image as fallback.
            if bbox is None:
                if args.fallback_full_image:
                    new_row = dict(row)
                    new_row["original_image_path"] = str(image_path)
                    new_row["image_path"] = str(image_path)
                    new_row["crop_status"] = "no_bbox_used_full_image"
                    fout.write(json.dumps(new_row) + "\n")
                    fallback += 1
                else:
                    skipped += 1
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                img_w, img_h = img.size

                x1, y1, x2, y2 = convert_bbox_to_xyxy(
                    bbox,
                    img_w,
                    img_h,
                    args.bbox_format,
                )

                x1, y1, x2, y2 = add_padding(
                    x1,
                    y1,
                    x2,
                    y2,
                    img_w,
                    img_h,
                    args.pad,
                )

                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue

                crop = img.crop((x1, y1, x2, y2))

                out_dir = crop_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)

                crop_path = out_dir / f"{image_path.stem}_gtcrop.jpg"
                crop.save(crop_path, quality=95)

                new_row = dict(row)
                new_row["original_image_path"] = str(image_path)
                new_row["image_path"] = str(crop_path)
                new_row["crop_status"] = "gt_bbox_crop"
                new_row["crop_bbox_xyxy"] = [x1, y1, x2, y2]
                new_row["crop_pad_frac"] = args.pad

                fout.write(json.dumps(new_row) + "\n")
                cropped += 1

            except Exception as e:
                print("ERROR:", image_path, e)
                skipped += 1

    print("Done.")
    print("Input metadata:", metadata_path)
    print("Output metadata:", out_metadata)
    print("Crop dir:", crop_dir)
    print("Total rows:", total)
    print("Cropped:", cropped)
    print("Fallback full image:", fallback)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
