#!/usr/bin/env python3
"""
Add Jetson tegrastats metrics to results/experiment_log.csv.

This script:
1. Reads results/experiment_log.csv.
2. Finds matching tegrastats logs in results/logs/ by experiment_id.
3. Calls src/summarize_tegrastats.py to save a readable summary.
4. Parses the tegrastats log into machine-readable fields:
   RAM, SWAP, GPU util, CPU util, VDD_IN power, temperature.
5. Writes results/experiment_log_with_tegrastats.csv by default,
   or overwrites experiment_log.csv with --in-place.

Recommended log naming:
  results/logs/tegrastats_qwen3vl4b_self_q4_k_m_mmprojq8_95.txt
  results/logs/tegrastats_qwen3vl4b_self_q5_k_m_mmprojq8_95.txt
  results/logs/tegrastats_llava15_7b_q4_cli_balanced_95.txt

Manual mapping example:
  python src/add_tegrastats_to_experiment_log.py \
    --map qwen3vl4b_self_q4_k_m_mmprojq8_95=results/logs/tegrastats_qwen3vl4b_q4_95.txt
"""

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_LOG_DIRS = [
    Path("results/logs"),
    Path("results"),
]

EXTRA_COLUMNS = [
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
    "avg_temp_c",
    "peak_temp_c",
    "cold_start_j_per_image",
]


def mean_or_blank(values):
    return statistics.mean(values) if values else ""


def max_or_blank(values):
    return max(values) if values else ""


def round_or_blank(value, ndigits=4):
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), ndigits)
    except Exception:
        return ""


def parse_boolish(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float_or_none(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def parse_tegrastats_log(path):
    ram_used_mb = []
    ram_total_mb = []
    swap_used_mb = []
    swap_total_mb = []
    gpu_util_pct = []
    cpu_utils = []
    vdd_in_mw = []
    temps_c = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Example: RAM 3902/7620MB
            m = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
            if m:
                ram_used_mb.append(int(m.group(1)))
                ram_total_mb.append(int(m.group(2)))

            # Example: SWAP 0/3810MB
            m = re.search(r"\bSWAP\s+(\d+)/(\d+)MB", line)
            if m:
                swap_used_mb.append(int(m.group(1)))
                swap_total_mb.append(int(m.group(2)))

            # Example: GR3D_FREQ 99%
            m = re.search(r"\bGR3D_FREQ\s+(\d+)%", line)
            if m:
                gpu_util_pct.append(int(m.group(1)))

            # Example: CPU [5%@729,3%@729,off,off,4%@729,2%@729]
            m = re.search(r"\bCPU\s+\[([^\]]+)\]", line)
            if m:
                for part in m.group(1).split(","):
                    part = part.strip()
                    if part == "off":
                        continue
                    cm = re.search(r"(\d+)%@", part)
                    if cm:
                        cpu_utils.append(int(cm.group(1)))

            # Example: VDD_IN 7587mW/7520mW
            # Use the first number as instantaneous power.
            m = re.search(r"\bVDD_IN\s+(\d+)mW", line)
            if m:
                vdd_in_mw.append(int(m.group(1)))

            # Example sensors: CPU@45.5C GPU@46C tj@47.5C
            for tm in re.finditer(r"@([\d.]+)C", line):
                temps_c.append(float(tm.group(1)))

    avg_power_w = mean_or_blank(vdd_in_mw)
    peak_power_w = max_or_blank(vdd_in_mw)

    if avg_power_w != "":
        avg_power_w = avg_power_w / 1000.0
    if peak_power_w != "":
        peak_power_w = peak_power_w / 1000.0

    return {
        "tegrastats_samples": len(ram_used_mb),
        "avg_ram_mb": round_or_blank(mean_or_blank(ram_used_mb), 2),
        "peak_ram_mb": round_or_blank(max_or_blank(ram_used_mb), 2),
        "ram_total_mb": round_or_blank(max_or_blank(ram_total_mb), 2),
        "avg_swap_mb": round_or_blank(mean_or_blank(swap_used_mb), 2),
        "peak_swap_mb": round_or_blank(max_or_blank(swap_used_mb), 2),
        "swap_total_mb": round_or_blank(max_or_blank(swap_total_mb), 2),
        "avg_gpu_util_pct": round_or_blank(mean_or_blank(gpu_util_pct), 2),
        "peak_gpu_util_pct": round_or_blank(max_or_blank(gpu_util_pct), 2),
        "avg_cpu_util_pct": round_or_blank(mean_or_blank(cpu_utils), 2),
        "peak_cpu_util_pct": round_or_blank(max_or_blank(cpu_utils), 2),
        "avg_power_w": round_or_blank(avg_power_w, 4),
        "peak_power_w": round_or_blank(peak_power_w, 4),
        "avg_temp_c": round_or_blank(mean_or_blank(temps_c), 2),
        "peak_temp_c": round_or_blank(max_or_blank(temps_c), 2),
    }


def canonical_tokens(text):
    text = str(text).lower()
    text = re.sub(r"\.(csv|txt|log)$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    drop = {
        "results",
        "logs",
        "log",
        "txt",
        "csv",
        "tegrastats",
        "power",
        "run",
        "eval",
        "balanced",
    }

    return [t for t in text.split() if t and t not in drop]


def match_score(experiment_id, log_path):
    exp_tokens = set(canonical_tokens(experiment_id))
    log_tokens = set(canonical_tokens(log_path.stem))

    if not exp_tokens or not log_tokens:
        return 0

    score = len(exp_tokens & log_tokens)

    exp_norm = "".join(canonical_tokens(experiment_id))
    log_norm = "".join(canonical_tokens(log_path.stem))

    if exp_norm and exp_norm in log_norm:
        score += 10
    if log_norm and log_norm in exp_norm:
        score += 5

    # Reward matching sample count tokens like 19 or 95.
    for n in ["19", "95"]:
        if n in exp_tokens and n in log_tokens:
            score += 2

    # Reward quantization match.
    for q in ["q3", "q4", "q5", "q6", "q8", "f16"]:
        if q in exp_tokens and q in log_tokens:
            score += 2

    return score


def collect_log_files(log_dirs):
    logs = []
    for d in log_dirs:
        if not d.exists():
            continue
        logs.extend(d.glob("*.txt"))
        logs.extend(d.glob("*.log"))
    return sorted(set(logs))


def parse_map_args(map_args):
    manual = {}
    for item in map_args:
        if "=" not in item:
            raise ValueError(f"Bad --map value {item!r}. Expected experiment_id=path/to/log.txt")
        k, v = item.split("=", 1)
        manual[k.strip()] = Path(v.strip())
    return manual


def find_matching_log(experiment_id, logs, manual, min_score):
    if experiment_id in manual:
        return manual[experiment_id], "manual", 999

    candidates = []
    for log in logs:
        score = match_score(experiment_id, log)
        if score >= min_score:
            candidates.append((score, log))

    candidates.sort(reverse=True, key=lambda x: x[0])

    if not candidates:
        return None, "none", 0

    if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
        return candidates[0][1], "tie_best_guess", candidates[0][0]

    return candidates[0][1], "auto", candidates[0][0]


def run_existing_summarizer(log_path, out_dir, experiment_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_id}_tegrastats_summary.txt"

    cmd = [
        sys.executable,
        "src/summarize_tegrastats.py",
        str(log_path),
    ]

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    out_path.write_text(proc.stdout, encoding="utf-8")
    return out_path, proc.returncode


def read_experiment_log(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_experiment_log(path, rows, fieldnames):
    for col in EXTRA_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-log", default="results/experiment_log.csv")
    parser.add_argument("--out", default="results/experiment_log_with_tegrastats.csv")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--log-dir", action="append", default=None)
    parser.add_argument("--map", action="append", default=[], help="experiment_id=path/to/tegrastats.log")
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-dir", default="results/tegrastats_summaries")
    args = parser.parse_args()

    experiment_log = Path(args.experiment_log)
    output_path = experiment_log if args.in_place else Path(args.out)

    if args.log_dir:
        log_dirs = [Path(x) for x in args.log_dir]
    else:
        log_dirs = DEFAULT_LOG_DIRS

    manual = parse_map_args(args.map)
    logs = collect_log_files(log_dirs)

    rows, fieldnames = read_experiment_log(experiment_log)

    print(f"Experiment log: {experiment_log}")
    print(f"Rows: {len(rows)}")
    print(f"Log dirs: {', '.join(str(d) for d in log_dirs)}")
    print(f"Found tegrastats logs: {len(logs)}")

    matched = 0

    for row in rows:
        experiment_id = row.get("experiment_id", "").strip()
        if not experiment_id:
            print("SKIP: row has no experiment_id")
            continue

        log_path, mode, score = find_matching_log(experiment_id, logs, manual, args.min_score)

        if log_path is None:
            print(f"NO LOG: {experiment_id}")
            continue

        if not log_path.exists():
            print(f"MISSING MANUAL LOG: {experiment_id} -> {log_path}")
            continue

        print(f"MATCH [{mode}, score={score}]: {experiment_id} -> {log_path}")

        if args.dry_run:
            continue

        stats = parse_tegrastats_log(log_path)

        row["tegrastats_log"] = str(log_path)
        for k, v in stats.items():
            row[k] = v

        # Existing experiment_log.csv has avg_wall_s, so use that with avg VDD_IN.
        avg_wall_s = parse_float_or_none(row.get("avg_wall_s"))
        avg_power_w = parse_float_or_none(row.get("avg_power_w"))

        if avg_wall_s is not None and avg_power_w is not None:
            row["cold_start_j_per_image"] = round_or_blank(avg_wall_s * avg_power_w, 4)

        summary_path, rc = run_existing_summarizer(
            log_path=log_path,
            out_dir=Path(args.summary_dir),
            experiment_id=experiment_id,
        )

        if rc != 0:
            print(f"  WARNING: summarize_tegrastats.py returned {rc}; saved output to {summary_path}")
        else:
            print(f"  saved readable summary: {summary_path}")

        matched += 1

    if args.dry_run:
        print("\nDry run only. No files written.")
        return

    write_experiment_log(output_path, rows, fieldnames)
    print(f"\nUpdated {matched}/{len(rows)} rows")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
