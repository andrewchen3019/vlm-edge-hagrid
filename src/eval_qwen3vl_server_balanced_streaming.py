#!/usr/bin/env python3

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
from typing import Any

import requests


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

# Compile these once instead of sorting labels and compiling regexes for every
# prediction. Longest labels are checked first so, for example,
# "peace_inverted" is not reduced to "peace".
LABEL_PATTERNS = tuple(
    (
        label,
        re.compile(
            rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])"
        ),
    )
    for label in sorted(LABELS, key=len, reverse=True)
)

IMAGE_FIELDS = (
    "image_path",
    "path",
    "image",
    "file_path",
    "filepath",
    "file_name",
    "filename",
)

LABEL_FIELDS = (
    "label",
    "class",
    "gesture",
    "category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-VL through a persistent llama-server. "
            "Each image is read, Base64-encoded, and serialized immediately "
            "before its timed HTTP request, so preprocessing is excluded "
            "without retaining every encoded image in memory."
        )
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to HaGRID metadata.jsonl.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/qwen_hagrid_label.txt",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--model",
        default="qwen3vl4b-q4",
        help="The alias passed to llama-server with --alias.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Images selected per label. Set to 0 to use every metadata row.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of excluded warm-up requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Controls the evaluation order.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle the selected evaluation rows.",
    )
    parser.add_argument(
        "--image-field",
        default=None,
        help="Override automatic image-path field detection.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Override automatic label field detection.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help=(
            "Flush the output CSV after this many measured images. "
            "Use 1 for maximum crash protection. Default: 10."
        ),
    )

    args = parser.parse_args()

    if args.per_class < 0:
        parser.error("--per-class must be 0 or greater")

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than 0")

    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    if args.warmup < 0:
        parser.error("--warmup must be 0 or greater")

    if args.flush_every <= 0:
        parser.error("--flush-every must be greater than 0")

    return args


def detect_field(
    row: dict[str, Any],
    requested: str | None,
    candidates: tuple[str, ...],
    field_description: str,
) -> str:
    if requested:
        if requested not in row:
            raise KeyError(
                f"Requested {field_description} field "
                f"{requested!r} is missing from metadata row."
            )
        return requested

    for field in candidates:
        if field in row:
            return field

    raise KeyError(
        f"Could not detect {field_description} field. "
        f"Available fields: {sorted(row.keys())}"
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
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise TypeError(
                    f"Metadata row {line_number} is not a JSON object."
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No metadata rows found in {path}")

    return rows


def resolve_image_path(
    raw_path: str,
    metadata_path: Path,
) -> Path:
    supplied = Path(raw_path).expanduser()

    candidates = [
        supplied,
        Path.cwd() / supplied,
        metadata_path.parent / supplied,
    ]

    # A row may contain only images/label/name.jpg while the command
    # is launched from another directory.
    if supplied.parts and supplied.parts[0] != "images":
        candidates.append(metadata_path.parent / "images" / supplied)

    checked: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in checked:
            continue

        checked.add(candidate)

        if candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Could not resolve image path {raw_path!r}.\nChecked:\n{formatted}"
    )


def select_balanced_rows(
    rows: list[dict[str, Any]],
    image_field: str,
    label_field: str,
    per_class: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, str]]:
    """Select only the two metadata values needed by the evaluator.

    The original implementation copied every full metadata dictionary. This
    version stores only the true label and image-path string, reducing Python
    object allocation and memory traffic during setup.
    """

    cleaned: list[dict[str, str]] = []

    for row in rows:
        label = str(row[label_field]).strip()

        if label not in LABELS:
            continue

        cleaned.append(
            {
                "true_label": label,
                "image_value": str(row[image_field]),
            }
        )

    if per_class <= 0:
        selected = cleaned
    else:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

        for row in cleaned:
            grouped[row["true_label"]].append(row)

        missing = [
            label
            for label in LABELS
            if len(grouped[label]) < per_class
        ]

        if missing:
            details = ", ".join(
                f"{label}={len(grouped[label])}"
                for label in missing
            )
            raise ValueError(
                f"Not enough images for --per-class {per_class}: {details}"
            )

        selected: list[dict[str, str]] = []

        # Select the first N metadata rows from each class. This keeps image
        # selection deterministic across quantized-model runs.
        for label in LABELS:
            selected.extend(grouped[label][:per_class])

    if shuffle:
        random.Random(seed).shuffle(selected)

    return selected


def image_to_data_url(image_path: Path) -> str:
    """Read and Base64-encode one image before its measured HTTP request."""

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
    """Build one complete serialized request immediately before timing.

    Image reading, Base64 encoding, payload construction, and JSON
    serialization occur before request_prediction starts its timer.
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Compact separators reduce request size and JSON parsing work without
    # changing the payload semantics.
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def prepare_evaluation_rows(
    selected_rows: list[dict[str, str]],
    metadata_path: Path,
) -> tuple[list[dict[str, Any]], float]:
    """Resolve all image paths without loading or encoding image contents."""

    prepared_rows: list[dict[str, Any]] = []
    total = len(selected_rows)

    start = time.perf_counter()

    print("Resolving image paths...")

    for index, row in enumerate(selected_rows, start=1):
        image_path = resolve_image_path(
            row["image_value"],
            metadata_path,
        )

        prepared_rows.append(
            {
                "true_label": row["true_label"],
                "image_path": image_path,
            }
        )

        if index == total or index % 25 == 0:
            print(f"  Resolved {index}/{total} images")

    elapsed_s = time.perf_counter() - start
    return prepared_rows, elapsed_s


def extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")

    if not choices:
        raise ValueError("Response contains no choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])

        return "\n".join(parts)

    return str(content)


def normalize_prediction(raw_text: str) -> str:
    text = raw_text.strip().lower()

    # Remove common formatting around a one-word answer.
    text = text.replace("`", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    text = text.strip(" \t\r\n.,:;!?[](){}")

    if text in LABELS:
        return text

    matches: list[str] = []

    for label, pattern in LABEL_PATTERNS:
        if pattern.search(text):
            matches.append(label)

    # LABEL_PATTERNS contains each label only once, so duplicates are not
    # expected. Retain the one-match rule to reject ambiguous answers.
    if len(matches) == 1:
        return matches[0]

    return "INVALID"


def request_prediction(
    session: requests.Session,
    endpoint_url: str,
    request_body: bytes,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    """Send one pre-serialized request and measure only request wall time.

    The timer excludes image disk I/O, Base64 encoding, payload construction,
    and client-side JSON serialization. It includes socket transfer, server
    processing, inference, and response transfer.
    """

    start = time.perf_counter()

    response = session.post(
        endpoint_url,
        data=request_body,
        timeout=timeout,
    )

    wall_s = time.perf_counter() - start

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Server returned non-JSON response "
            f"(HTTP {response.status_code}):\n{response.text[:2000]}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Server returned HTTP {response.status_code}:\n"
            f"{json.dumps(response_json, indent=2)[:4000]}"
        )

    return response_json, wall_s


def wait_for_server(
    session: requests.Session,
    base_url: str,
    timeout_s: float = 300.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"

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
        f"llama-server did not become ready within {timeout_s:.0f} seconds."
    )


def percentile_95(values: list[float]) -> float:
    if not values:
        return math.nan

    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def calculate_macro_f1(
    true_labels: list[str],
    predicted_labels: list[str],
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
        f1 = 0.0 if denominator == 0 else (2 * tp) / denominator
        scores.append(f1)

    return sum(scores) / len(scores)


def main() -> int:
    args = parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    output_path = Path(args.out).expanduser()
    prompt_path = Path(args.prompt_file).expanduser()

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)

    prompt = prompt_path.read_text(encoding="utf-8").strip()

    # The OpenAI-compatible chat endpoint inserts the image from the
    # image_url block. A manual marker causes the media-marker mismatch.
    if "<__media__>" in prompt:
        raise ValueError(
            "Remove <__media__> from the prompt file. "
            "The server API inserts the image automatically."
        )

    rows = load_metadata(metadata_path)

    image_field = detect_field(
        rows[0],
        args.image_field,
        IMAGE_FIELDS,
        "image path",
    )

    label_field = detect_field(
        rows[0],
        args.label_field,
        LABEL_FIELDS,
        "label",
    )

    selected_rows = select_balanced_rows(
        rows=rows,
        image_field=image_field,
        label_field=label_field,
        per_class=args.per_class,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    # The complete metadata list is no longer needed. Delete the reference
    # before request preloading so Python can reclaim it if memory is tight.
    del rows

    if not selected_rows:
        raise ValueError("No evaluation images were selected.")

    prepared_rows, path_resolution_s = prepare_evaluation_rows(
        selected_rows=selected_rows,
        metadata_path=metadata_path,
    )

    del selected_rows
    endpoint_url = (
        f"{args.base_url.rstrip('/')}/v1/chat/completions"
    )

    print()
    print(f"Metadata:        {metadata_path}")
    print(f"Image field:     {image_field}")
    print(f"Label field:     {label_field}")
    print(f"Images:          {len(prepared_rows)}")
    print(f"Model alias:     {args.model}")
    print(f"Server:          {args.base_url}")
    print(f"Output:          {output_path}")
    print(f"Path resolution: {path_resolution_s:.3f} s")
    print(
        "For each image, disk I/O, Base64 encoding, payload creation, and "
        "JSON serialization occur immediately before the timer starts."
    )
    print(
        "Measured wall time includes only the HTTP request, server processing, "
        "inference, and response transfer."
    )
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

    session = requests.Session()

    # Avoid proxy-environment checks for each localhost request and preserve a
    # single persistent TCP connection to llama-server.
    session.trust_env = False
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )

    wait_for_server(session, args.base_url)

    # Warm up the model, CUDA kernels, image encoder, and server slot.
    # The warm-up request is built immediately before the untimed request.
    warmup_row = prepared_rows[0]
    warmup_image: Path = warmup_row["image_path"]

    for warmup_index in range(args.warmup):
        print(
            f"Warm-up {warmup_index + 1}/{args.warmup}: "
            f"{warmup_image.name}"
        )

        warmup_body = build_request_body(
            model=args.model,
            image_path=warmup_image,
            prompt=prompt,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        request_prediction(
            session=session,
            endpoint_url=endpoint_url,
            request_body=warmup_body,
            timeout=args.timeout,
        )

        del warmup_body

    if args.warmup:
        print()

    results: list[dict[str, Any]] = []

    try:
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()

            total = len(prepared_rows)

            for index, row in enumerate(prepared_rows, start=1):
                image_path: Path = row["image_path"]
                true_label: str = row["true_label"]

                raw_text = ""
                predicted_label = "INVALID"
                error_text = ""
                wall_s = math.nan
                timings: dict[str, Any] = {}

                request_body: bytes | None = None
                response_json: dict[str, Any] | None = None

                try:
                    # This work is intentionally outside request_prediction's
                    # timer and only one encoded request is retained at a time.
                    request_body = build_request_body(
                        model=args.model,
                        image_path=image_path,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                    )

                    response_json, wall_s = request_prediction(
                        session=session,
                        endpoint_url=endpoint_url,
                        request_body=request_body,
                        timeout=args.timeout,
                    )

                    raw_text = extract_text(response_json)
                    predicted_label = normalize_prediction(raw_text)
                    timings = response_json.get("timings") or {}

                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"

                finally:
                    # CPython releases these large temporary objects immediately
                    # when their reference counts reach zero.
                    del request_body
                    del response_json

                valid = predicted_label in LABELS
                correct = valid and predicted_label == true_label

                result = {
                    "index": index,
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "valid": int(valid),
                    "correct": int(correct),
                    "wall_s": (
                        f"{wall_s:.6f}"
                        if math.isfinite(wall_s)
                        else ""
                    ),
                    "prompt_ms": timings.get("prompt_ms", ""),
                    "predicted_ms": timings.get("predicted_ms", ""),
                    "prompt_tokens": timings.get("prompt_n", ""),
                    "cached_tokens": timings.get("cache_n", ""),
                    "predicted_tokens": timings.get("predicted_n", ""),
                    "raw_response": raw_text,
                    "error": error_text,
                }

                writer.writerow(result)

                # Avoid a filesystem flush after every image. The file is
                # still flushed periodically and on normal context-manager
                # exit. It is also flushed after the final image.
                if index % args.flush_every == 0 or index == total:
                    csv_file.flush()

                # This is outside the latency timer. Refcounting normally frees
                # image/request objects immediately; periodic GC handles cycles.
                if index % 25 == 0:
                    gc.collect()

                results.append(result)

                status = "OK" if correct else "WRONG"

                if not valid:
                    status = "INVALID"

                wall_display = (
                    f"{wall_s:.2f}s"
                    if math.isfinite(wall_s)
                    else "ERROR"
                )

                print(
                    f"[{index:03d}/{total:03d}] "
                    f"true={true_label:<17} "
                    f"pred={predicted_label:<17} "
                    f"{status:<7} "
                    f"{wall_display}"
                )

                if error_text:
                    print(f"    {error_text[:500]}")

    finally:
        session.close()

    true_labels = [row["true_label"] for row in results]
    predicted_labels = [row["predicted_label"] for row in results]

    latencies = [
        float(row["wall_s"])
        for row in results
        if row["wall_s"]
    ]

    correct_count = sum(int(row["correct"]) for row in results)
    invalid_count = sum(not int(row["valid"]) for row in results)

    accuracy = correct_count / len(results)
    macro_f1 = calculate_macro_f1(
        true_labels,
        predicted_labels,
    )

    print()
    print("Evaluation complete")
    print(f"Images:         {len(results)}")
    print(f"Correct:        {correct_count}")
    print(f"Invalid:        {invalid_count}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}")
    print(f"Path resolve:   {path_resolution_s:.3f} s (excluded)")

    if latencies:
        print(f"Average wall:   {statistics.mean(latencies):.3f} s")
        print(f"Median wall:    {statistics.median(latencies):.3f} s")
        print(f"P95 wall:       {percentile_95(latencies):.3f} s")
        print(f"Measured total: {sum(latencies):.3f} s")
        print(
            f"Throughput:     "
            f"{len(latencies) / sum(latencies):.3f} images/s"
        )

    print(f"CSV:            {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
