#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix

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

parser = argparse.ArgumentParser()
parser.add_argument("results_csv")
parser.add_argument("--out", default="confusion.csv")
args = parser.parse_args()

df = pd.read_csv(args.results_csv)

cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"],
    labels=LABELS,
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS,
)

cm_df.to_csv(args.out)

print(f"Saved confusion matrix to {args.out}")