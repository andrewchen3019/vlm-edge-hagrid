#!/usr/bin/env python3
"""
Rebuild results/experiment_log.csv for 95-image runs only.

This script combines:
- result CSV summarization
- tegrastats parsing
- optional readable tegrastats summaries using src/summarize_tegrastats.py
- markdown table creation without tabulate

It only includes eval CSVs with exactly 95 rows/images.
"""

from pathlib import Path
from datetime import date
import argparse
import csv
import re
import statistics
import subprocess
import sys


RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
OUT_CSV = RESULTS_DIR / "experiment_log.csv"
OUT_MD = RESULTS_DIR / "experiment_log.md"
TEGRASUM_DIR = RESULTS_DIR / "tegrastats_summaries"

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
    "tegrastats_log",
    "tegrastats_samples",
    "avg_ram_mb",
    "peak_ram_mb",
    "ram_total_mb",
    "avg_swap_mb",
    "peak_swap_mb",
    "swap_total_mb",
    "avg_gpu_util_pct",
    "peak_gpu_util_pct",
    "avg_cpu_util_pct",
    "peak_cpu_util_pct",
    "avg_power_w",
    "peak_power_w",
    "cold_start_j_per_image",
    "avg_temp_c",
    "peak_temp_c",
    "notes",
]

# Edit these if your experiment_id filenames differ.
DEFAULT_TEGRATS_MAP = {
    "qwen3vl4b_q4_cli_balanced_95": "results/logs/tegrastats_qwen3vl4b_q4_cli_balanced_95.txt",
    "qwen3vl4b_self_q4_k_m_mmprojq8_95": "results/logs/tegrastats_qwen3vl4b_q4_cli_balanced_95.txt",

    "qwen3vl4b_self_q5_k_m_mmprojq8_95": "results/logs/tegrastats_qwen3vl4b_q5km_95.txt",
    "qwen3vl4b_q5km_95": "results/logs/tegrastats_qwen3vl4b_q5km_95.txt",

    "qwen3vl4b_self_q8_0_mmprojq8_95": "results/logs/tegrastats_qwen3vl4b_q8_95.txt",
    "qwen3vl4b_q8_95": "results/logs/tegrastats_qwen3vl4b_q8_95.txt",
}


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def boolish(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return statistics.median(vals)


def quantile(values, q):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = round((len(vals) - 1) * q)
    return vals[idx]


def round_blank(value, ndigits=4):
    if value is None or value == "":
        return ""
    try:
        return round(float(value), ndigits)
    except Exception:
        return ""


def get_col(rows, names):
    for name in names:
        if rows and name in rows[0]:
            return [to_float(r.get(name)) for r in rows]
    return []


def macro_f1(y_true, y_pred):
    f1s = []

    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def should_skip_csv(path):
    name = path.name.lower()

    if name in {"experiment_log.csv", "experiment_log_with_tegrastats.csv"}:
        return True
    if "experiment_log" in name:
        return True
    if "confusion" in name:
        return True
    if "mistakes" in name:
        return True

    return False


def infer_quant(stem):
    s = stem.lower()

    if "f16" in s:
        return "F16"
    if "q8_0" in s or "_q8_" in s:
        return "Q8_0"
    if "q6_k" in s:
        return "Q6_K"
    if "q5_k_m" in s or "q5km" in s:
        return "Q5_K_M"
    if "q4_k_m" in s or "_q4_" in s:
        return "Q4_K_M"
    if "q3_k_m" in s:
        return "Q3_K_M"

    return ""


def infer_metadata(stem, num_images):
    s = stem.lower()

    row = {c: "" for c in COLUMNS}

    row["experiment_id"] = stem
    row["date"] = str(date.today())
    row["input_type"] = "full_image"
    row["dataset_split"] = "HaGRID_day1_balanced"
    row["num_images"] = num_images
    row["per_class"] = 5 if num_images == 95 else ""
    row["temp"] = 0
    row["fit"] = "off"

    if "llava" in s:
        row["model"] = "LLaVA-1.5-7B"
        row["model_family"] = "LLaVA"
        row["main_quant"] = "Q4_K_M"
        row["main_model_file"] = "models/llava-v1.5-7b-second-state/llava-v1.5-7b-Q4_K_M.gguf"
        row["mmproj_file"] = "models/llava-v1.5-7b-second-state/llava-v1.5-7b-mmproj-model-f16.gguf"
        row["mmproj_quant"] = "F16"
        row["ctx"] = 2048
        row["ngl"] = 10
        row["max_tokens"] = 8
        row["no_mmproj_offload"] = True
        row["power_mode"] = "15W"
        row["jetson_clocks"] = True
        row["notes"] = "LLaVA Q4 CLI-per-image baseline, 95-image balanced set"

    elif "qwen" in s or "q4" in s or "q5" in s or "q8" in s:
        row["model"] = "Qwen3-VL-4B-Instruct"
        row["model_family"] = "Qwen3-VL"
        row["main_quant"] = infer_quant(stem)
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
        else:
            row["power_mode"] = "15W"
            row["jetson_clocks"] = True
            row["notes"] = f"Qwen3-VL {row['main_quant']} CLI-per-image run, 95-image balanced set"

        if "self" in s and row["main_quant"]:
            row["main_model_file"] = f"models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-{row['main_quant']}-self.gguf"
        elif row["main_quant"] == "F16":
            row["main_model_file"] = "models/qwen3-vl-4b-instruct-gguf/Qwen3VL-4B-Instruct-F16.gguf"
        elif row["main_quant"]:
            row["main_model_file"] = f"models/qwen3-vl-4b-instruct-gguf/Qwen3VL-4B-Instruct-{row['main_quant']}.gguf"

    return row


def summarize_eval_csv(path):
    rows = read_csv_rows(path)

    if len(rows) != 95:
        return None

    if not rows or "true" not in rows[0] or "pred" not in rows[0]:
        return None

    y_true = [r.get("true", "") for r in rows]
    y_pred = [r.get("pred", "") for r in rows]

    if "correct" in rows[0]:
        correct = [boolish(r.get("correct")) for r in rows]
        acc = sum(correct) / len(correct)
    else:
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(rows)

    if "invalid" in rows[0]:
        invalid_count = sum(boolish(r.get("invalid")) for r in rows)
    else:
        invalid_count = sum(p == "INVALID" for p in y_pred)

    wall = get_col(rows, ["wall_s", "wall_s_persistent", "latency_s", "elapsed_s"])
    mtmd = get_col(rows, ["mtmd_encode_ms"])

    row = infer_metadata(path.stem, len(rows))

    row["accuracy"] = round_blank(acc, 4)
    row["macro_f1"] = round_blank(macro_f1(y_true, y_pred), 4)
    row["invalid_count"] = invalid_count
    row["invalid_rate"] = round_blank(invalid_count / len(rows), 4)

    if wall:
        row["avg_wall_s"] = round_blank(mean(wall), 4)
        row["median_wall_s"] = round_blank(median(wall), 4)
        row["p95_wall_s"] = round_blank(quantile(wall, 0.95), 4)

    if mtmd:
        row["avg_mtmd_encode_ms"] = round_blank(mean(mtmd), 2)
        row["p95_mtmd_encode_ms"] = round_blank(quantile(mtmd, 0.95), 2)

    return row


def parse_tegrastats(path):
    ram_used = []
    ram_total = []
    swap_used = []
    swap_total = []
    gpu_util = []
    cpu_util = []
    power_mw = []
    temps = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
            if m:
                ram_used.append(int(m.group(1)))
                ram_total.append(int(m.group(2)))

            m = re.search(r"\bSWAP\s+(\d+)/(\d+)MB", line)
            if m:
                swap_used.append(int(m.group(1)))
                swap_total.append(int(m.group(2)))

            m = re.search(r"\bGR3D_FREQ\s+(\d+)%", line)
            if m:
                gpu_util.append(int(m.group(1)))

            m = re.search(r"\bCPU\s+\[([^\]]+)\]", line)
            if m:
                for part in m.group(1).split(","):
                    cm = re.search(r"(\d+)%@", part)
                    if cm:
                        cpu_util.append(int(cm.group(1)))

            m = re.search(r"\bVDD_IN\s+(\d+)mW", line)
            if m:
                power_mw.append(int(m.group(1)))

            for tm in re.finditer(r"@([\d.]+)C", line):
                temps.append(float(tm.group(1)))

    avg_power_w = mean(power_mw)
    peak_power_w = max(power_mw) if power_mw else None

    if avg_power_w is not None:
        avg_power_w /= 1000.0
    if peak_power_w is not None:
        peak_power_w /= 1000.0

    return {
        "tegrastats_log": str(path),
        "tegrastats_samples": len(ram_used),
        "avg_ram_mb": round_blank(mean(ram_used), 2),
        "peak_ram_mb": round_blank(max(ram_used) if ram_used else None, 2),
        "ram_total_mb": round_blank(max(ram_total) if ram_total else None, 2),
        "avg_swap_mb": round_blank(mean(swap_used), 2),
        "peak_swap_mb": round_blank(max(swap_used) if swap_used else None, 2),
        "swap_total_mb": round_blank(max(swap_total) if swap_total else None, 2),
        "avg_gpu_util_pct": round_blank(mean(gpu_util), 2),
        "peak_gpu_util_pct": round_blank(max(gpu_util) if gpu_util else None, 2),
        "avg_cpu_util_pct": round_blank(mean(cpu_util), 2),
        "peak_cpu_util_pct": round_blank(max(cpu_util) if cpu_util else None, 2),
        "avg_power_w": round_blank(avg_power_w, 4),
        "peak_power_w": round_blank(peak_power_w, 4),
        "avg_temp_c": round_blank(mean(temps), 2),
        "peak_temp_c": round_blank(max(temps) if temps else None, 2),
    }


def run_existing_tegrastats_summary(experiment_id, log_path):
    script = Path("src/summarize_tegrastats.py")
    if not script.exists():
        return

    TEGRASUM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEGRASUM_DIR / f"{experiment_id}_tegrastats_summary.txt"

    proc = subprocess.run(
        [sys.executable, str(script), str(log_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    out_path.write_text(proc.stdout, encoding="utf-8")


def parse_manual_maps(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --map value: {item}")
        k, v = item.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def apply_tegrastats(row, mapping):
    exp_id = row["experiment_id"]

    log_str = mapping.get(exp_id)
    if not log_str:
        return row

    log_path = Path(log_str)
    if not log_path.exists():
        print(f"WARNING: mapped tegrastats log does not exist for {exp_id}: {log_path}")
        return row

    stats = parse_tegrastats(log_path)
    row.update(stats)

    avg_wall_s = to_float(row.get("avg_wall_s"))
    avg_power_w = to_float(row.get("avg_power_w"))
    if avg_wall_s is not None and avg_power_w is not None:
        row["cold_start_j_per_image"] = round_blank(avg_wall_s * avg_power_w, 4)

    run_existing_tegrastats_summary(exp_id, log_path)

    return row


def escape_md(value):
    s = "" if value is None else str(value)
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ")
    return s


def write_markdown_table(path, rows):
    md_cols = [
        "experiment_id",
        "model",
        "main_quant",
        "num_images",
        "accuracy",
        "macro_f1",
        "invalid_count",
        "avg_wall_s",
        "p95_wall_s",
        "avg_ram_mb",
        "peak_ram_mb",
        "avg_gpu_util_pct",
        "avg_power_w",
        "peak_power_w",
        "avg_temp_c",
        "peak_temp_c",
        "cold_start_j_per_image",
        "notes",
    ]

    lines = []
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(escape_md(row.get(c, "")) for c in md_cols) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--no-default-tegrastats-map", action="store_true")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Manual mapping: experiment_id=path/to/tegrastats_log.txt",
    )
    args = parser.parse_args()

    mapping = {}
    if not args.no_default_tegrastats_map:
        mapping.update(DEFAULT_TEGRATS_MAP)
    mapping.update(parse_manual_maps(args.map))

    all_rows = []

    for path in sorted(RESULTS_DIR.glob("*.csv")):
        if should_skip_csv(path):
            continue

        row = summarize_eval_csv(path)
        if row is None:
            continue

        row = apply_tegrastats(row, mapping)
        all_rows.append(row)
        print(f"Included 95-image run: {path}")

    all_rows = sorted(all_rows, key=lambda r: r["experiment_id"])

    write_csv_rows(Path(args.out_csv), all_rows, COLUMNS)
    write_markdown_table(Path(args.out_md), all_rows)

    print()
    print(f"Wrote {args.out_csv} with {len(all_rows)} 95-image runs")
    print(f"Wrote {args.out_md}")
    print(f"Wrote readable tegrastats summaries to {TEGRASUM_DIR} when logs were mapped")


if __name__ == "__main__":
    main()
