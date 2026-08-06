#!/usr/bin/env python3
"""Evaluate Qwen3-VL through a persistent llama-server instance.

Image loading, Base64 encoding, request construction, and JSON serialization
happen before the request timer starts. The recorded ``wall_s`` therefore
measures HTTP transfer, server processing, inference, and response transfer,
not local image preprocessing.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import json
import math
import mimetypes
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import requests

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

LABEL_PATTERNS = tuple(
    (
        label,
        re.compile(rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])"),
    )
    for label in sorted(LABELS, key=len, reverse=True)
)

IMAGE_FIELDS: tuple[str, ...] = (
    "image_path",
    "path",
    "image",
    "file_path",
    "filepath",
    "file_name",
    "filename",
)

LABEL_FIELDS: tuple[str, ...] = (
    "label",
    "class",
    "gesture",
    "category",
)


class EvaluationError(RuntimeError):
    """Raised when the benchmark cannot be executed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-VL through llama-server using the final "
            "380-image HaGRID benchmark settings by default."
        )
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Exact alias supplied to llama-server with --alias.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/qwen_hagrid_label.txt"),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Images selected per class. Use 0 for every matching row.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warm-up requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Preserve class-grouped metadata order.",
    )
    parser.add_argument("--image-field", default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-path-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Write image paths relative to this directory when possible. "
            "Default: current repository directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV.",
    )
    args = parser.parse_args()

    if args.per_class < 0:
        parser.error("--per-class must be zero or greater")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.server_ready_timeout <= 0:
        parser.error("--server-ready-timeout must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.flush_every <= 0:
        parser.error("--flush-every must be greater than zero")

    return args


def detect_field(
    row: dict[str, Any],
    requested: str | None,
    candidates: Sequence[str],
    description: str,
) -> str:
    if requested:
        if requested not in row:
            raise EvaluationError(
                f"Requested {description} field {requested!r} is missing. "
                f"Available fields: {sorted(row)}"
            )
        return requested

    for candidate in candidates:
        if candidate in row:
            return candidate

    raise EvaluationError(
        f"Could not detect {description} field. Available fields: {sorted(row)}"
    )


def load_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvaluationError(
                    f"Metadata row {line_number} is not a JSON object"
                )
            rows.append(row)

    if not rows:
        raise EvaluationError(f"No metadata rows found in {path}")
    return rows


def resolve_image_path(raw_path: str, metadata_path: Path) -> Path:
    supplied = Path(raw_path).expanduser()
    candidates = [
        supplied,
        Path.cwd() / supplied,
        metadata_path.parent / supplied,
    ]

    if supplied.parts and supplied.parts[0] != "images":
        candidates.append(metadata_path.parent / "images" / supplied)

    checked: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if resolved.is_file():
            return resolved

    detail = "\n".join(f"  - {candidate}" for candidate in checked)
    raise FileNotFoundError(
        f"Could not resolve image path {raw_path!r}. Checked:\n{detail}"
    )


def portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def select_balanced_rows(
    rows: Sequence[dict[str, Any]],
    image_field: str,
    label_field: str,
    per_class: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        label = str(row[label_field]).strip()
        if label not in LABELS:
            continue
        grouped[label].append(
            {
                "true_label": label,
                "image_value": str(row[image_field]),
            }
        )

    if per_class == 0:
        selected = [
            item
            for label in LABELS
            for item in grouped.get(label, [])
        ]
    else:
        missing = [
            f"{label}={len(grouped.get(label, []))}"
            for label in LABELS
            if len(grouped.get(label, [])) < per_class
        ]
        if missing:
            raise EvaluationError(
                f"Not enough images for --per-class {per_class}: "
                + ", ".join(missing)
            )
        selected = [
            item
            for label in LABELS
            for item in grouped[label][:per_class]
        ]

    if shuffle:
        random.Random(seed).shuffle(selected)
    return selected


def prepare_rows(
    selected_rows: Sequence[dict[str, str]],
    metadata_path: Path,
    output_path_root: Path,
) -> tuple[list[dict[str, Any]], float]:
    prepared: list[dict[str, Any]] = []
    start = time.perf_counter()

    for index, row in enumerate(selected_rows, start=1):
        resolved = resolve_image_path(row["image_value"], metadata_path)
        prepared.append(
            {
                "true_label": row["true_label"],
                "image_path": resolved,
                "csv_image_path": portable_path(resolved, output_path_root),
            }
        )
        if index % 25 == 0 or index == len(selected_rows):
            print(f"  Resolved {index}/{len(selected_rows)} images")

    return prepared, time.perf_counter() - start


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_request_body(
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> bytes:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not choices:
        raise EvaluationError("Response contains no choices")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def normalize_prediction(raw_text: str) -> str:
    text = raw_text.strip().lower()
    text = text.replace("`", "").replace('"', "").replace("'", "")
    text = text.strip(" \t\r\n.,:;!?[](){}")

    if text in LABELS:
        return text

    matches = [
        label for label, pattern in LABEL_PATTERNS if pattern.search(text)
    ]
    return matches[0] if len(matches) == 1 else "INVALID"


def request_prediction(
    session: requests.Session,
    endpoint_url: str,
    request_body: bytes,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = session.post(endpoint_url, data=request_body, timeout=timeout)
    wall_s = time.perf_counter() - start

    try:
        response_json = response.json()
    except ValueError as exc:
        raise EvaluationError(
            f"Non-JSON response (HTTP {response.status_code}): "
            f"{response.text[:1000]}"
        ) from exc

    if not response.ok:
        raise EvaluationError(
            f"HTTP {response.status_code}: "
            f"{json.dumps(response_json, ensure_ascii=False)[:2000]}"
        )
    return response_json, wall_s


def wait_for_server(
    session: requests.Session,
    base_url: str,
    timeout_s: float,
) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    print(f"Waiting for {health_url} ...")

    while time.monotonic() < deadline:
        try:
            response = session.get(health_url, timeout=5)
            if response.status_code == 200:
                print("Server is ready.")
                return
        except requests.RequestException:
            pass
        time.sleep(2)

    raise TimeoutError(
        f"llama-server did not become ready within {timeout_s:.0f} seconds"
    )


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def calculate_macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> float:
    scores: list[float] = []
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
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return statistics.fmean(scores)


def main() -> int:
    args = parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    prompt_path = args.prompt_file.expanduser().resolve()
    output_path = args.out.expanduser()
    output_path_root = args.output_path_root.expanduser().resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    if output_path.exists() and not args.overwrite:
        raise EvaluationError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise EvaluationError("Prompt file is empty")
    if "<__media__>" in prompt:
        raise EvaluationError(
            "Remove <__media__> from the prompt; the API image_url block "
            "inserts the image automatically."
        )

    metadata_rows = load_metadata(metadata_path)
    image_field = detect_field(
        metadata_rows[0], args.image_field, IMAGE_FIELDS, "image path"
    )
    label_field = detect_field(
        metadata_rows[0], args.label_field, LABEL_FIELDS, "label"
    )
    selected = select_balanced_rows(
        rows=metadata_rows,
        image_field=image_field,
        label_field=label_field,
        per_class=args.per_class,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    del metadata_rows

    if not selected:
        raise EvaluationError("No evaluation images were selected")

    print("Resolving image paths...")
    prepared, path_resolution_s = prepare_rows(
        selected, metadata_path, output_path_root
    )
    del selected

    print()
    print(f"Metadata:        {metadata_path}")
    print(f"Images:          {len(prepared)}")
    print(f"Model alias:     {args.model}")
    print(f"Server:          {args.base_url}")
    print(f"Output:          {output_path}")
    print(f"Path resolution: {path_resolution_s:.3f} s (excluded)")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "image_path",
        "true_label",
        "predicted_label",
        "valid",
        "correct",
        "wall_s",
        "prompt_ms",
        "predicted_ms",
        "prompt_tokens",
        "cached_tokens",
        "predicted_tokens",
        "raw_response",
        "error",
    ]

    endpoint_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )

    results: list[dict[str, Any]] = []
    try:
        wait_for_server(session, args.base_url, args.server_ready_timeout)

        warmup_image: Path = prepared[0]["image_path"]
        for warmup_index in range(args.warmup):
            print(
                f"Warm-up {warmup_index + 1}/{args.warmup}: "
                f"{warmup_image.name}"
            )
            body = build_request_body(
                args.model,
                warmup_image,
                prompt,
                args.max_tokens,
                args.seed,
            )
            request_prediction(session, endpoint_url, body, args.timeout)
            del body

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            handle.flush()

            total = len(prepared)
            for index, row in enumerate(prepared, start=1):
                image_path: Path = row["image_path"]
                true_label: str = row["true_label"]
                raw_text = ""
                predicted_label = "INVALID"
                error_text = ""
                wall_s = math.nan
                timings: dict[str, Any] = {}

                try:
                    body = build_request_body(
                        args.model,
                        image_path,
                        prompt,
                        args.max_tokens,
                        args.seed,
                    )
                    response_json, wall_s = request_prediction(
                        session, endpoint_url, body, args.timeout
                    )
                    raw_text = extract_text(response_json)
                    predicted_label = normalize_prediction(raw_text)
                    timings = response_json.get("timings") or {}
                except Exception as exc:  # preserve a row for failed requests
                    error_text = f"{type(exc).__name__}: {exc}"
                finally:
                    body = None
                    response_json = None

                valid = predicted_label in LABELS
                correct = valid and predicted_label == true_label
                result = {
                    "index": index,
                    "image_path": row["csv_image_path"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "valid": int(valid),
                    "correct": int(correct),
                    "wall_s": f"{wall_s:.6f}" if math.isfinite(wall_s) else "",
                    "prompt_ms": timings.get("prompt_ms", ""),
                    "predicted_ms": timings.get("predicted_ms", ""),
                    "prompt_tokens": timings.get("prompt_n", ""),
                    "cached_tokens": timings.get("cache_n", ""),
                    "predicted_tokens": timings.get("predicted_n", ""),
                    "raw_response": raw_text,
                    "error": error_text,
                }
                writer.writerow(result)
                results.append(result)

                if index % args.flush_every == 0 or index == total:
                    handle.flush()
                if index % 25 == 0:
                    gc.collect()

                status = "OK" if correct else "WRONG"
                if not valid:
                    status = "INVALID"
                wall_display = (
                    f"{wall_s:.2f}s" if math.isfinite(wall_s) else "ERROR"
                )
                print(
                    f"[{index:03d}/{total:03d}] "
                    f"true={true_label:<17} "
                    f"pred={predicted_label:<17} "
                    f"{status:<7} {wall_display}"
                )
                if error_text:
                    print(f"    {error_text[:500]}")
    finally:
        session.close()

    true_labels = [row["true_label"] for row in results]
    predicted_labels = [row["predicted_label"] for row in results]
    latencies = [
        float(row["wall_s"]) for row in results if row["wall_s"]
    ]
    correct_count = sum(int(row["correct"]) for row in results)
    invalid_count = sum(not int(row["valid"]) for row in results)

    print()
    print("Evaluation complete")
    print(f"Images:         {len(results)}")
    print(f"Correct:        {correct_count}")
    print(f"Invalid:        {invalid_count}")
    print(f"Accuracy:       {correct_count / len(results):.4f}")
    print(
        f"Macro-F1:       "
        f"{calculate_macro_f1(true_labels, predicted_labels):.4f}"
    )
    if latencies:
        print(f"Average wall:   {statistics.fmean(latencies):.3f} s")
        print(f"Median wall:    {statistics.median(latencies):.3f} s")
        print(f"P95 wall:       {percentile(latencies, 0.95):.3f} s")
        print(f"Measured total: {sum(latencies):.3f} s")
        print(f"Throughput:     {len(latencies) / sum(latencies):.3f} images/s")
    print(f"CSV:            {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
