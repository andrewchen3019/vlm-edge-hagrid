import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to evaluation CSV")
    parser.add_argument("--save-confusion", action="store_true", help="Save confusion matrix CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    required = ["true", "pred"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "invalid" not in df.columns:
        df["invalid"] = df["pred"].eq("INVALID")

    if "correct" not in df.columns:
        df["correct"] = df["true"].eq(df["pred"])

    print("=" * 80)
    print("FILE:", csv_path)
    print("Images:", len(df))

    print("\nQUALITY")
    print("Correct:", int(df["correct"].sum()), "/", len(df))
    print("Accuracy:", accuracy_score(df["true"], df["pred"]))
    print("Macro-F1:", f1_score(df["true"], df["pred"], average="macro", zero_division=0))
    print("Invalid count:", int(df["invalid"].sum()))
    print("Invalid rate:", df["invalid"].mean())

    print("\nLATENCY")
    timing_cols = [
        "wall_s",
        "mtmd_encode_ms",
        "prompt_eval_ms",
        "decode_eval_ms",
        "total_ms",
    ]

    for col in timing_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                print(f"{col}: no valid values")
                continue

            print(f"{col} avg:", vals.mean())
            print(f"{col} median:", vals.median())
            print(f"{col} p95:", vals.quantile(0.95))

    print("\nTRUE LABEL COUNTS")
    print(df["true"].value_counts().sort_index())

    print("\nPREDICTION COUNTS")
    print(df["pred"].value_counts().sort_index())

    print("\nMOST COMMON CONFUSIONS")
    mistakes = df[df["true"] != df["pred"]]
    if len(mistakes) == 0:
        print("No mistakes.")
    else:
        print(
            mistakes.groupby(["true", "pred"])
            .size()
            .sort_values(ascending=False)
            .head(20)
        )

    print("\nPER-CLASS REPORT")
    print(classification_report(df["true"], df["pred"], zero_division=0))

    print("\nCONFUSION MATRIX")
    labels = sorted(set(df["true"]) | set(df["pred"]))
    cm = confusion_matrix(df["true"], df["pred"], labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    print(cm_df)

    if args.save_confusion:
        out_path = csv_path.with_name(csv_path.stem + "_confusion.csv")
        cm_df.to_csv(out_path)
        print("\nSaved confusion matrix:", out_path)


if __name__ == "__main__":
    main()
