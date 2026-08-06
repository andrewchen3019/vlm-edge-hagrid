import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("confusion_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    csv_path = Path(args.confusion_csv)
    cm_df = pd.read_csv(csv_path, index_col=0)

    labels = list(cm_df.index)
    cm = cm_df.values

    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

    fig_size = max(10, len(labels) * 0.6)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation="nearest")
    plt.title(csv_path.stem)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            if value != 0:
                plt.text(j, i, str(value), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
