#!/usr/bin/env python3
"""Create a 19-class confusion-matrix CSV from one prediction CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABELS: tuple[str, ...] = (
    "call", "no_gesture", "dislike", "fist", "four", "like", "mute",
    "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("confusion.csv"))
    args = parser.parse_args()

    counts = {true: {predicted: 0 for predicted in LABELS} for true in LABELS}
    with args.results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"true_label", "predicted_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required)}")
        for row in reader:
            true = row["true_label"]
            predicted = row["predicted_label"]
            if true not in counts:
                raise ValueError(f"Unexpected true label: {true}")
            if predicted in LABELS:
                counts[true][predicted] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *LABELS])
        for true in LABELS:
            writer.writerow([true, *(counts[true][predicted] for predicted in LABELS)])

    print(f"Saved confusion matrix to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
