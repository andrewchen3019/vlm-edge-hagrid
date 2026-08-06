#!/usr/bin/env python3
"""Compile the final Qwen3-VL quantization benchmark statistics.

This script reads the five final per-image prediction CSVs and their matching
NVIDIA tegrastats logs, then writes one comparison CSV containing accuracy,
latency, throughput, utilization, memory, power, energy, temperature,
per-class accuracy, and confusion statistics.

The default paths match this repository's final-results layout:

    results/image_results/
    results/tegrastat_logs/
    results/summary/quantization_comparison.csv

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


LABELS: tuple[str, ...] = (
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
)

# Conservative semantic grouping used as a secondary, application-tolerant
# accuracy metric. Palm and stop remain separate because they are distinct
# target gestures, even though the model often confuses them.
GESTURE_FAMILY_MAP: dict[str, str] = {
    "peace": "peace_family",
    "peace_inverted": "peace_family",
    "stop": "stop_family",
    "stop_inverted": "stop_family",
    "two_up": "two_up_family",
    "two_up_inverted": "two_up_family",
    "three": "three_family",
    "three2": "three_family",
}

GESTURE_FAMILY_RULE = (
    "Merged peace+peace_inverted; stop+stop_inverted; "
    "two_up+two_up_inverted; three+three2. "
    "Palm and stop remain separate."
)

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "image_path",
    "true_label",
    "predicted_label",
    "valid",
    "correct",
    "wall_s",
    "prompt_ms",
    "predicted_ms",
    "predicted_tokens",
    "error",
)

TIMESTAMP_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%")
GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
GPU_TEMP_RE = re.compile(r"gpu@([0-9.]+)C")
CPU_TEMP_RE = re.compile(r"cpu@([0-9.]+)C")
POWER_RE = re.compile(r"VDD_IN\s+(\d+)mW")


@dataclass(frozen=True)
class ModelSpec:
    model: str
    quantization: str
    prediction_filename: str
    tegrastats_filename: str
    # Validated sample indices reproduce the final report's active and idle
    # windows. They can be replaced by automatic detection with --window-mode
    # auto when analyzing newly collected logs.
    validated_active_start: int
    validated_idle_start: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model="Q8",
        quantization="Q8_0",
        prediction_filename="qwen3vl4b_q8_336px_380_fullgpu.csv",
        tegrastats_filename="qwen3vl4b_q8_336px_380_fullgpu_tegrastats.txt",
        validated_active_start=10,
        validated_idle_start=0,
    ),
    ModelSpec(
        model="Q6",
        quantization="Q6_K",
        prediction_filename="qwen3vl4b_q6_336px_380_fullgpu.csv",
        tegrastats_filename="qwen3vl4b_q6_336px_380_fullgpu_tegrastats.txt",
        validated_active_start=35,
        validated_idle_start=15,
    ),
    ModelSpec(
        model="Q5",
        quantization="Q5_K_M",
        prediction_filename="qwen3vl4b_q5_336px_380_fullgpu.csv",
        tegrastats_filename="qwen3vl4b_q5_336px_380_fullgpu_tegrastats.txt",
        validated_active_start=9,
        validated_idle_start=0,
    ),
    ModelSpec(
        model="Q4",
        quantization="Q4_K_M",
        prediction_filename="qwen3vl4b_q4_336px_380_fullgpu.csv",
        tegrastats_filename="qwen3vl4b_q4_336px_380_fullgpu_tegrastats.txt",
        validated_active_start=10,
        validated_idle_start=0,
    ),
    ModelSpec(
        model="Q3",
        quantization="Q3_K_M",
        prediction_filename="qwen3vl4b_q3_336px_380_fullgpu.csv",
        tegrastats_filename="qwen3vl4b_q3_336px_380_fullgpu_tegrastats.txt",
        validated_active_start=17,
        validated_idle_start=0,
    ),
)


@dataclass(frozen=True)
class TegrastatsSample:
    timestamp: datetime
    ram_used_mb: int
    ram_total_mb: int
    gpu_util_pct: int
    emc_util_pct: int | None
    board_power_w: float
    gpu_temp_c: float | None
    cpu_temp_c: float | None
    cpu_values: tuple[int, ...]


@dataclass(frozen=True)
class TegrastatsWindow:
    active_start: int
    active_end: int
    idle_start: int
    idle_end: int


class AnalysisError(RuntimeError):
    """Raised when an input file cannot support a valid final analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the final Qwen3-VL Q8/Q6/Q5/Q4/Q3 benchmark "
            "statistics from prediction CSVs and tegrastats logs."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/image_results"),
        help="Directory containing the five final prediction CSVs.",
    )
    parser.add_argument(
        "--tegrastats-dir",
        type=Path,
        default=Path("results/tegrastat_logs"),
        help="Directory containing the five final tegrastats logs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary/quantization_comparison.csv"),
        help="Destination comparison CSV.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("validated", "auto"),
        default="validated",
        help=(
            "Use the manually validated final-run tegrastats windows, or "
            "automatically choose the GPU-active cluster whose duration is "
            "closest to the summed request wall time."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="HaGRID balanced 19-class subset",
        help="Dataset description written to the output CSV.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=336,
        help="Maximum image side used in the final benchmark.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON copy of the compiled records.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model status messages.",
    )
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be greater than zero")

    return args


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise AnalysisError(f"Cannot parse boolean value: {value!r}")


def parse_float(row: dict[str, str], column: str, source: Path) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(
            f"Invalid numeric value for {column!r} in {source}: "
            f"{row.get(column)!r}"
        ) from exc


def parse_int(row: dict[str, str], column: str, source: Path) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(
            f"Invalid integer value for {column!r} in {source}: "
            f"{row.get(column)!r}"
        ) from exc


def load_predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisError(f"Prediction CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisError(f"Prediction CSV has no header: {path}")

        missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise AnalysisError(
                f"Prediction CSV {path} is missing columns: {', '.join(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise AnalysisError(f"Prediction CSV has no rows: {path}")

    invalid_true = sorted(
        {row["true_label"] for row in rows if row["true_label"] not in LABELS}
    )
    if invalid_true:
        raise AnalysisError(
            f"Unexpected true labels in {path}: {', '.join(invalid_true)}"
        )

    image_paths = [row["image_path"] for row in rows]
    if len(set(image_paths)) != len(image_paths):
        raise AnalysisError(f"Duplicate image paths found in {path}")

    return rows


def parse_cpu_values(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in text.split(","):
        match = re.match(r"\s*(\d+)%", item)
        if match:
            values.append(int(match.group(1)))
    return tuple(values)


def parse_tegrastats(path: Path) -> list[TegrastatsSample]:
    if not path.is_file():
        raise AnalysisError(f"tegrastats log not found: {path}")

    samples: list[TegrastatsSample] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            timestamp_match = TIMESTAMP_RE.search(line)
            ram_match = RAM_RE.search(line)
            gpu_match = GPU_RE.search(line)
            power_match = POWER_RE.search(line)

            # Ignore non-tegrastats lines rather than failing on server output
            # accidentally redirected into the same file.
            if not (timestamp_match and ram_match and gpu_match and power_match):
                continue

            try:
                timestamp = datetime.strptime(
                    timestamp_match.group(1), "%m-%d-%Y %H:%M:%S"
                )
            except ValueError as exc:
                raise AnalysisError(
                    f"Invalid timestamp in {path}:{line_number}"
                ) from exc

            emc_match = EMC_RE.search(line)
            gpu_temp_match = GPU_TEMP_RE.search(line)
            cpu_temp_match = CPU_TEMP_RE.search(line)
            cpu_match = CPU_RE.search(line)

            samples.append(
                TegrastatsSample(
                    timestamp=timestamp,
                    ram_used_mb=int(ram_match.group(1)),
                    ram_total_mb=int(ram_match.group(2)),
                    gpu_util_pct=int(gpu_match.group(1)),
                    emc_util_pct=(
                        int(emc_match.group(1)) if emc_match else None
                    ),
                    board_power_w=int(power_match.group(1)) / 1000.0,
                    gpu_temp_c=(
                        float(gpu_temp_match.group(1))
                        if gpu_temp_match
                        else None
                    ),
                    cpu_temp_c=(
                        float(cpu_temp_match.group(1))
                        if cpu_temp_match
                        else None
                    ),
                    cpu_values=(
                        parse_cpu_values(cpu_match.group(1))
                        if cpu_match
                        else ()
                    ),
                )
            )

    if not samples:
        raise AnalysisError(f"No valid tegrastats samples found in {path}")

    return samples


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile, matching the final report."""
    if not values:
        return math.nan
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    lower_weight = upper - position
    upper_weight = position - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def safe_mean(values: Iterable[float | int | None], name: str) -> float:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        raise AnalysisError(f"No values available for {name}")
    return statistics.fmean(cleaned)


def calculate_macro_metrics(
    true_labels: Sequence[str], predicted_labels: Sequence[str]
) -> tuple[float, float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for label in LABELS:
        tp = sum(
            true == label and predicted == label
            for true, predicted in zip(true_labels, predicted_labels)
        )
        fp = sum(
            true != label and predicted == label
            for true, predicted in zip(true_labels, predicted_labels)
        )
        fn = sum(
            true == label and predicted != label
            for true, predicted in zip(true_labels, predicted_labels)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return (
        statistics.fmean(precisions),
        statistics.fmean(recalls),
        statistics.fmean(f1_scores),
    )


def gesture_family(label: str) -> str:
    return GESTURE_FAMILY_MAP.get(label, label)


def positive_gpu_clusters(
    samples: Sequence[TegrastatsSample], max_zero_gap_samples: int = 3
) -> list[tuple[int, int]]:
    """Return GPU-active clusters while tolerating short request-boundary gaps."""
    positive_indices = [
        index for index, sample in enumerate(samples) if sample.gpu_util_pct > 0
    ]
    if not positive_indices:
        raise AnalysisError("tegrastats log contains no GPU-active samples")

    clusters: list[tuple[int, int]] = []
    start = previous = positive_indices[0]

    for index in positive_indices[1:]:
        # Difference of 4 means three zero samples occurred between positives.
        if index - previous > max_zero_gap_samples + 1:
            clusters.append((start, previous))
            start = index
        previous = index

    clusters.append((start, previous))
    return clusters


def choose_auto_window(
    samples: Sequence[TegrastatsSample], measured_wall_time_s: float
) -> TegrastatsWindow:
    clusters = positive_gpu_clusters(samples)

    def cluster_score(cluster: tuple[int, int]) -> tuple[float, float]:
        start, end = cluster
        duration = (samples[end].timestamp - samples[start].timestamp).total_seconds()
        # Primary score: closest to measured request time. Secondary: prefer
        # longer clusters when two candidates are similarly close.
        return (abs(duration - measured_wall_time_s), -duration)

    active_start, active_end = min(clusters, key=cluster_score)

    # Use the contiguous GPU-idle region immediately before the selected run.
    # If a startup/load burst occurred, begin after its last positive sample.
    previous_positive = max(
        (
            index
            for index in range(active_start)
            if samples[index].gpu_util_pct > 0
        ),
        default=-1,
    )
    idle_start = previous_positive + 1
    idle_end = active_start - 1

    # Discard a single elevated transition sample after model loading when
    # enough stable idle samples remain. This reproduces the intent of the
    # manually validated Q6 idle window without model-specific knowledge.
    while idle_end - idle_start + 1 > 5:
        window_values = [
            sample.board_power_w for sample in samples[idle_start : idle_end + 1]
        ]
        median_power = statistics.median(window_values)
        if samples[idle_start].board_power_w <= median_power * 1.20:
            break
        idle_start += 1

    if idle_start > idle_end:
        raise AnalysisError("Could not find an idle window before inference")

    return TegrastatsWindow(
        active_start=active_start,
        active_end=active_end,
        idle_start=idle_start,
        idle_end=idle_end,
    )


def choose_validated_window(
    spec: ModelSpec,
    samples: Sequence[TegrastatsSample],
) -> TegrastatsWindow:
    if spec.validated_active_start >= len(samples):
        raise AnalysisError(
            f"Validated active start {spec.validated_active_start} is outside "
            f"the {len(samples)} samples for {spec.model}"
        )

    active_end = max(
        (
            index
            for index in range(spec.validated_active_start, len(samples))
            if samples[index].gpu_util_pct > 0
        ),
        default=-1,
    )
    if active_end < spec.validated_active_start:
        raise AnalysisError(f"No GPU-active samples found for {spec.model}")

    idle_end = spec.validated_active_start - 1
    if spec.validated_idle_start > idle_end:
        raise AnalysisError(f"Invalid validated idle window for {spec.model}")

    return TegrastatsWindow(
        active_start=spec.validated_active_start,
        active_end=active_end,
        idle_start=spec.validated_idle_start,
        idle_end=idle_end,
    )


def validate_window_duration(
    spec: ModelSpec,
    samples: Sequence[TegrastatsSample],
    window: TegrastatsWindow,
    measured_wall_time_s: float,
) -> str | None:
    observed_duration_s = (
        samples[window.active_end].timestamp
        - samples[window.active_start].timestamp
    ).total_seconds()
    difference_s = observed_duration_s - measured_wall_time_s

    if abs(difference_s) <= max(10.0, measured_wall_time_s * 0.03):
        return None

    return (
        f"{spec.model}: tegrastats active duration is {observed_duration_s:.1f}s, "
        f"but summed request wall time is {measured_wall_time_s:.1f}s "
        f"(difference {difference_s:+.1f}s). Check the selected window."
    )


def top_confusions(
    true_labels: Sequence[str], predicted_labels: Sequence[str], limit: int = 10
) -> str:
    counts = Counter(
        (true, predicted)
        for true, predicted in zip(true_labels, predicted_labels)
        if true != predicted
    )
    return "; ".join(
        f"{true}->{predicted}:{count}"
        for (true, predicted), count in counts.most_common(limit)
    )


def analyze_model(
    spec: ModelSpec,
    prediction_path: Path,
    tegrastats_path: Path,
    dataset_name: str,
    resolution: int,
    window_mode: str,
) -> tuple[dict[str, Any], str | None]:
    rows = load_predictions(prediction_path)
    samples = parse_tegrastats(tegrastats_path)

    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]

    wall_times = [parse_float(row, "wall_s", prediction_path) for row in rows]
    prompt_times_ms = [
        parse_float(row, "prompt_ms", prediction_path) for row in rows
    ]
    generation_times_ms = [
        parse_float(row, "predicted_ms", prediction_path) for row in rows
    ]
    server_times = [
        (prompt_ms + generation_ms) / 1000.0
        for prompt_ms, generation_ms in zip(
            prompt_times_ms, generation_times_ms
        )
    ]
    generated_tokens = [
        parse_int(row, "predicted_tokens", prediction_path) for row in rows
    ]

    total_wall_time_s = sum(wall_times)

    if window_mode == "validated":
        window = choose_validated_window(spec, samples)
    else:
        window = choose_auto_window(samples, total_wall_time_s)

    active = samples[window.active_start : window.active_end + 1]
    idle = samples[window.idle_start : window.idle_end + 1]

    warning = validate_window_duration(
        spec, samples, window, total_wall_time_s
    )

    exact_correct = sum(
        true == predicted
        for true, predicted in zip(true_labels, predicted_labels)
    )
    gesture_family_correct = sum(
        gesture_family(true) == gesture_family(predicted)
        for true, predicted in zip(true_labels, predicted_labels)
    )

    valid_responses = sum(parse_bool(row["valid"]) for row in rows)
    errors_or_timeouts = sum(bool(row["error"].strip()) for row in rows)

    macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(
        true_labels, predicted_labels
    )

    class_counts = Counter(true_labels)
    class_correct = Counter(
        true
        for true, predicted in zip(true_labels, predicted_labels)
        if true == predicted
    )

    unique_class_counts = set(class_counts.values())
    images_per_class: int | str
    if len(unique_class_counts) == 1 and len(class_counts) == len(LABELS):
        images_per_class = next(iter(unique_class_counts))
    else:
        images_per_class = "mixed"

    cpu_mean_per_sample = [
        statistics.fmean(sample.cpu_values)
        for sample in active
        if sample.cpu_values
    ]
    busiest_core_per_sample = [
        max(sample.cpu_values) for sample in active if sample.cpu_values
    ]

    avg_power_w = safe_mean(
        (sample.board_power_w for sample in active), "active board power"
    )
    idle_power_w = safe_mean(
        (sample.board_power_w for sample in idle), "idle board power"
    )

    token_counts = Counter(generated_tokens)
    ram_total_values = {sample.ram_total_mb for sample in active}
    if len(ram_total_values) != 1:
        raise AnalysisError(
            f"RAM total changed within {spec.model} active window: "
            f"{sorted(ram_total_values)}"
        )
    ram_total_mb = next(iter(ram_total_values))
    peak_ram_used_mb = max(sample.ram_used_mb for sample in active)

    record: dict[str, Any] = {
        "model": spec.model,
        "quantization": spec.quantization,
        "dataset": dataset_name,
        "image_resolution_px": resolution,
        "num_classes": len(LABELS),
        "images_per_class": images_per_class,
        "images_evaluated": len(rows),
        "unique_images": len({row["image_path"] for row in rows}),
        "valid_responses": valid_responses,
        "invalid_responses": len(rows) - valid_responses,
        "errors_or_timeouts": errors_or_timeouts,
        "exact_correct": exact_correct,
        "exact_accuracy_pct": 100.0 * exact_correct / len(rows),
        "macro_precision": macro_precision,
        "macro_precision_pct": 100.0 * macro_precision,
        "macro_recall": macro_recall,
        "macro_recall_pct": 100.0 * macro_recall,
        "macro_f1": macro_f1,
        "macro_f1_pct": 100.0 * macro_f1,
        "gesture_family_correct": gesture_family_correct,
        "gesture_family_accuracy_pct": (
            100.0 * gesture_family_correct / len(rows)
        ),
        "gesture_family_grouping_rule": GESTURE_FAMILY_RULE,
        "min_wall_latency_s": min(wall_times),
        "avg_wall_latency_s": statistics.fmean(wall_times),
        "wall_latency_std_s": statistics.stdev(wall_times),
        "median_wall_latency_s": statistics.median(wall_times),
        "p90_wall_latency_s": percentile(wall_times, 0.90),
        "p95_wall_latency_s": percentile(wall_times, 0.95),
        "p99_wall_latency_s": percentile(wall_times, 0.99),
        "max_wall_latency_s": max(wall_times),
        "avg_server_latency_s": statistics.fmean(server_times),
        "median_server_latency_s": statistics.median(server_times),
        "p95_server_latency_s": percentile(server_times, 0.95),
        "max_server_latency_s": max(server_times),
        "avg_client_http_overhead_s": statistics.fmean(
            wall - server
            for wall, server in zip(wall_times, server_times)
        ),
        "wall_throughput_images_per_s": len(rows) / total_wall_time_s,
        "server_throughput_images_per_s": (
            1.0 / statistics.fmean(server_times)
        ),
        "total_measured_wall_time_s": total_wall_time_s,
        "avg_prompt_time_ms": statistics.fmean(prompt_times_ms),
        "avg_generation_time_ms": statistics.fmean(generation_times_ms),
        "avg_generated_tokens": statistics.fmean(generated_tokens),
        "generated_2_token_responses": token_counts.get(2, 0),
        "generated_3_token_responses": token_counts.get(3, 0),
        "generated_4_token_responses": token_counts.get(4, 0),
        "generated_5_token_responses": token_counts.get(5, 0),
        "tegrastats_window_mode": window_mode,
        "tegrastats_active_start_index": window.active_start,
        "tegrastats_active_end_index": window.active_end,
        "tegrastats_active_start_time": samples[
            window.active_start
        ].timestamp.isoformat(),
        "tegrastats_active_end_time": samples[
            window.active_end
        ].timestamp.isoformat(),
        "tegrastats_active_samples": len(active),
        "tegrastats_active_duration_s": (
            samples[window.active_end].timestamp
            - samples[window.active_start].timestamp
        ).total_seconds(),
        "avg_gpu_util_pct": safe_mean(
            (sample.gpu_util_pct for sample in active), "GPU utilization"
        ),
        "pct_samples_gpu_at_least_90": (
            100.0
            * sum(sample.gpu_util_pct >= 90 for sample in active)
            / len(active)
        ),
        "peak_gpu_util_pct": max(sample.gpu_util_pct for sample in active),
        "avg_emc_util_pct": safe_mean(
            (sample.emc_util_pct for sample in active), "EMC utilization"
        ),
        "avg_cpu_util_per_core_pct": safe_mean(
            cpu_mean_per_sample, "CPU utilization"
        ),
        "avg_busiest_core_util_pct": safe_mean(
            busiest_core_per_sample, "busiest-core utilization"
        ),
        "peak_single_core_util_pct": max(busiest_core_per_sample),
        "ram_total_mb": ram_total_mb,
        "avg_ram_used_mb": safe_mean(
            (sample.ram_used_mb for sample in active), "RAM utilization"
        ),
        "peak_ram_used_mb": peak_ram_used_mb,
        "minimum_free_ram_mb": ram_total_mb - peak_ram_used_mb,
        "avg_board_power_w": avg_power_w,
        "peak_board_power_w": max(sample.board_power_w for sample in active),
        "idle_board_power_w": idle_power_w,
        "estimated_total_board_energy_wh": (
            avg_power_w * total_wall_time_s / 3600.0
        ),
        "estimated_energy_j_per_image": (
            avg_power_w * total_wall_time_s / len(rows)
        ),
        "incremental_energy_j_per_image_above_idle": (
            (avg_power_w - idle_power_w) * total_wall_time_s / len(rows)
        ),
        "avg_gpu_temperature_c": safe_mean(
            (sample.gpu_temp_c for sample in active), "GPU temperature"
        ),
        "peak_gpu_temperature_c": max(
            sample.gpu_temp_c
            for sample in active
            if sample.gpu_temp_c is not None
        ),
        "peak_cpu_temperature_c": max(
            sample.cpu_temp_c
            for sample in active
            if sample.cpu_temp_c is not None
        ),
        "top_10_confusions": top_confusions(true_labels, predicted_labels),
    }

    for label in LABELS:
        count = class_counts.get(label, 0)
        record[f"class_accuracy_{label}_pct"] = (
            100.0 * class_correct.get(label, 0) / count if count else math.nan
        )

    return record, warning


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        raise AnalysisError("No records were produced")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def print_summary(records: Sequence[dict[str, Any]]) -> None:
    print()
    print(
        f"{'Model':<6} {'Exact':>8} {'Family':>8} "
        f"{'Wall(s)':>9} {'Power(W)':>9} {'RAM peak':>10} {'J/image':>9}"
    )
    print("-" * 70)
    for record in records:
        print(
            f"{record['model']:<6} "
            f"{record['exact_accuracy_pct']:>7.2f}% "
            f"{record['gesture_family_accuracy_pct']:>7.2f}% "
            f"{record['avg_wall_latency_s']:>9.3f} "
            f"{record['avg_board_power_w']:>9.2f} "
            f"{record['peak_ram_used_mb']:>9.0f}M "
            f"{record['estimated_energy_j_per_image']:>9.2f}"
        )


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        for spec in MODEL_SPECS:
            prediction_path = args.predictions_dir / spec.prediction_filename
            tegrastats_path = args.tegrastats_dir / spec.tegrastats_filename

            if not args.quiet:
                print(f"Analyzing {spec.model} ({spec.quantization})...")
                print(f"  predictions: {prediction_path}")
                print(f"  tegrastats:  {tegrastats_path}")

            record, warning = analyze_model(
                spec=spec,
                prediction_path=prediction_path,
                tegrastats_path=tegrastats_path,
                dataset_name=args.dataset_name,
                resolution=args.resolution,
                window_mode=args.window_mode,
            )
            records.append(record)
            if warning:
                warnings.append(warning)

        write_csv(args.out, records)
        if args.json_out is not None:
            write_json(args.json_out, records)

    except (AnalysisError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print_summary(records)
        print(f"\nWrote {len(records)} model rows to {args.out}")
        if args.json_out is not None:
            print(f"Wrote JSON copy to {args.json_out}")

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
