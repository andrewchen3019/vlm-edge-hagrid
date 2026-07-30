from pathlib import Path
import re
import csv
import math
import pandas as pd
from datetime import date

RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "experiment_log.csv"
OUT_MD = RESULTS_DIR / "experiment_log.md"

COLUMNS = [
    "experiment_id",
    "date",
    "model",
    "model_family",
    "main_model_file",
    "main_quant",
    "mmproj_file",
    "mmproj_quant",
    "input_type",
    "dataset_split",
    "num_images",
    "per_class",
    "prompt_file",
    "ctx",
    "ngl",
    "max_tokens",
    "temp",
    "fit",
    "no_mmproj_offload",
    "power_mode",
    "jetson_clocks",
    "accuracy",
    "macro_f1",
    "invalid_count",
    "invalid_rate",
    "avg_wall_s",
    "median_wall_s",
    "p95_wall_s",
    "avg_mtmd_encode_ms",
    "p95_mtmd_encode_ms",
    "peak_ram_mb",
    "avg_ram_mb",
    "avg_gpu_util_pct",
    "peak_gpu_util_pct",
    "avg_power_w",
    "peak_power_w",
    "cold_start_j_per_image",
    "avg_temp_c",
    "peak_temp_c",
    "notes",
]

TRUE_LABELS = [
    "call", "no_gesture", "dislike", "fist", "four", "like", "mute",
    "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted"
]


def clean_float(x, ndigits=4):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
        return round(float(x), ndigits)
    except Exception:
        return ""


def get_numeric(df, names):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def macro_f1_from_labels(y_true, y_pred):
    # Macro-F1 over the 19 true HaGRID labels.
    # INVALID predictions count as false negatives for the true class,
    # but are not treated as an extra class.
    f1s = []
    for label in TRUE_LABELS:
        tp = sum((t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label and p != label) for t, p in zip(y_true, y_pred))

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


def infer_quant_from_name(stem):
    s = stem.lower()

    if "f16" in s:
        return "F16"
    if "q8_0" in s:
        return "Q8_0"
    if "q6_k" in s:
        return "Q6_K"
    if "q5_k_m" in s:
        return "Q5_K_M"
    if "q4_k_m" in s or "_q4_" in s or s.startswith("q4_"):
        return "Q4_K_M"
    if "q3_k_m" in s:
        return "Q3_K_M"

    return ""


def infer_model_file(stem, model_family, main_quant):
    s = stem.lower()

    if model_family == "LLaVA":
        return "models/llava-v1.5-7b-second-state/llava-v1.5-7b-Q4_K_M.gguf"

    if model_family == "Qwen3-VL":
        if "self" in s:
            return f"models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-{main_quant}-self.gguf"
        if main_quant == "F16":
            return "models/qwen3-vl-4b-instruct-gguf/Qwen3VL-4B-Instruct-F16.gguf"
        if main_quant:
            return f"models/qwen3-vl-4b-instruct-gguf/Qwen3VL-4B-Instruct-{main_quant}.gguf"

    return ""


def infer_metadata(stem, df):
    s = stem.lower()
    row = {c: "" for c in COLUMNS}

    row["experiment_id"] = stem
    row["date"] = str(date.today())
    row["input_type"] = "full_image"
    row["dataset_split"] = "HaGRID_day1_balanced"
    row["num_images"] = len(df)

    if len(df) % 19 == 0:
        row["per_class"] = len(df) // 19

    row["temp"] = 0
    row["fit"] = "off"

    if "llava" in s:
        row["model"] = "LLaVA-1.5-7B"
        row["model_family"] = "LLaVA"
        row["main_quant"] = "Q4_K_M"
        row["mmproj_file"] = "models/llava-v1.5-7b-second-state/llava-v1.5-7b-mmproj-model-f16.gguf"
        row["mmproj_quant"] = "F16"
        row["ctx"] = 2048
        row["ngl"] = 10
        row["max_tokens"] = 8
        row["no_mmproj_offload"] = True
        row["power_mode"] = "15W"
        row["jetson_clocks"] = True
        row["notes"] = "LLaVA Q4 CLI per-image baseline on HaGRID balanced set"

    elif "qwen" in s or s.startswith("q4_"):
        row["model"] = "Qwen3-VL-4B-Instruct"
        row["model_family"] = "Qwen3-VL"
        row["main_quant"] = infer_quant_from_name(stem)
        row["mmproj_file"] = "models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
        row["mmproj_quant"] = "Q8_0"
        row["ctx"] = 1024
        row["ngl"] = 10
        row["max_tokens"] = 16
        row["prompt_file"] = "prompts/qwen_hagrid_label.txt"
        row["no_mmproj_offload"] = True

        if "l40s" in s:
            row["power_mode"] = "L40S"
            row["jetson_clocks"] = ""
            row["ngl"] = 999
            row["notes"] = "Qwen3-VL F16 reference run on L40S server"
        elif "server" in s:
            row["notes"] = "Qwen3-VL Q4 llama-server persistent/server experiment"
            row["power_mode"] = "15W"
            row["jetson_clocks"] = True
        elif "self" in s:
            row["notes"] = f"Qwen3-VL self-quantized {row['main_quant']} CLI run with Q8 mmproj"
            row["power_mode"] = "15W"
            row["jetson_clocks"] = True
        else:
            row["notes"] = "Qwen3-VL CLI per-image baseline"
            row["power_mode"] = "15W"
            row["jetson_clocks"] = True

    row["main_model_file"] = infer_model_file(stem, row["model_family"], row["main_quant"])

    return row


def summarize_csv(path):
    df = pd.read_csv(path)
    stem = path.stem

    row = infer_metadata(stem, df)

    if "true" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"{path} does not look like an eval CSV: missing true/pred")

    y_true = df["true"].astype(str).tolist()
    y_pred = df["pred"].astype(str).tolist()

    if "correct" in df.columns:
        correct = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
        accuracy = correct.mean()
    else:
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    if "invalid" in df.columns:
        invalid = df["invalid"].astype(str).str.lower().isin(["true", "1", "yes"])
        invalid_count = int(invalid.sum())
    else:
        invalid_count = sum(p == "INVALID" for p in y_pred)

    row["accuracy"] = clean_float(accuracy, 4)
    row["macro_f1"] = clean_float(macro_f1_from_labels(y_true, y_pred), 4)
    row["invalid_count"] = invalid_count
    row["invalid_rate"] = clean_float(invalid_count / len(df), 4)

    wall = get_numeric(df, ["wall_s", "wall_s_persistent", "latency_s", "elapsed_s"])
    if wall is not None:
        row["avg_wall_s"] = clean_float(wall.mean(), 4)
        row["median_wall_s"] = clean_float(wall.median(), 4)
        row["p95_wall_s"] = clean_float(wall.quantile(0.95), 4)

    mtmd = get_numeric(df, ["mtmd_encode_ms"])
    if mtmd is not None:
        row["avg_mtmd_encode_ms"] = clean_float(mtmd.mean(), 2)
        row["p95_mtmd_encode_ms"] = clean_float(mtmd.quantile(0.95), 2)

    return row


def should_skip(path):
    name = path.name.lower()

    if name == "experiment_log.csv":
        return True
    if name.endswith("_confusion.csv"):
        return True
    if name.endswith("_mistakes.csv"):
        return True
    if "confusion" in name:
        return True
    if "experiment_log" in name:
        return True

    return False


def main():
    rows = []

    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if should_skip(path):
            continue

        try:
            row = summarize_csv(path)
            rows.append(row)
            print(f"OK: {path}")
        except Exception as e:
            print(f"SKIP/ERROR: {path}: {e}")

    rows = sorted(rows, key=lambda r: r["experiment_id"])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUT_CSV} with {len(rows)} runs")

    # Smaller readable markdown table
    md_cols = [
        "experiment_id",
        "model",
        "main_quant",
        "mmproj_quant",
        "num_images",
        "per_class",
        "accuracy",
        "macro_f1",
        "invalid_count",
        "avg_wall_s",
        "p95_wall_s",
        "avg_mtmd_encode_ms",
        "notes",
    ]

    md_df = pd.DataFrame(rows)[md_cols]
    OUT_MD.write_text(md_df.to_markdown(index=False), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
