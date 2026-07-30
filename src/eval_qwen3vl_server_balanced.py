#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
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
        description="Evaluate Qwen3-VL through a persistent llama-server."
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

    return parser.parse_args()


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
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []

    for row in rows:
        label = str(row[label_field]).strip()

        if label not in LABELS:
            continue

        copied = dict(row)
        copied["_true_label"] = label
        copied["_image_value"] = str(row[image_field])
        cleaned.append(copied)

    if per_class <= 0:
        selected = cleaned
    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in cleaned:
            grouped[row["_true_label"]].append(row)

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

        selected = []

        # Select the first N metadata rows from each class. This makes
        # image selection deterministic across quantized-model runs.
        for label in LABELS:
            selected.extend(grouped[label][:per_class])

    if shuffle:
        random.Random(seed).shuffle(selected)

    return selected


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


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

    # Search longest labels first so "peace_inverted" is not reduced
    # to "peace".
    matches: list[str] = []

    for label in sorted(LABELS, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])"

        if re.search(pattern, text):
            matches.append(label)

    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) == 1:
        return unique_matches[0]

    return "INVALID"


def request_prediction(
    session: requests.Session,
    base_url: str,
    model: str,
    image_data_url: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    seed: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
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

    start = time.perf_counter()

    response = session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
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

    selected = select_balanced_rows(
        rows=rows,
        image_field=image_field,
        label_field=label_field,
        per_class=args.per_class,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    resolved_rows: list[dict[str, Any]] = []

    for row in selected:
        copied = dict(row)
        copied["_resolved_image"] = resolve_image_path(
            copied["_image_value"],
            metadata_path,
        )
        resolved_rows.append(copied)

    print(f"Metadata:      {metadata_path}")
    print(f"Image field:   {image_field}")
    print(f"Label field:   {label_field}")
    print(f"Images:        {len(resolved_rows)}")
    print(f"Model alias:   {args.model}")
    print(f"Server:        {args.base_url}")
    print(f"Output:        {output_path}")
    print()

    if not resolved_rows:
        raise ValueError("No evaluation images were selected.")

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
    wait_for_server(session, args.base_url)

    # Warm up model, CUDA kernels, image encoder and server slot.

    print("Preloading and encoding images...")

    for row in resolved_rows:
        row["_image_data_url"] = image_to_data_url(row["_resolved_image"])

    warmup_image = resolved_rows[0]["_resolved_image"]

    for warmup_index in range(args.warmup):
        print(
            f"Warm-up {warmup_index + 1}/{args.warmup}: "
            f"{warmup_image.name}"
        )

        request_prediction(
            session=session,
            base_url=args.base_url,
            model=args.model,
            image_path=warmup_image,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=args.seed,
        )

    results: list[dict[str, Any]] = []

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        total = len(resolved_rows)

        for index, row in enumerate(resolved_rows, start=1):
            image_path: Path = row["_resolved_image"]
            true_label = row["_true_label"]

            raw_text = ""
            predicted_label = "INVALID"
            error_text = ""
            wall_s = math.nan
            timings: dict[str, Any] = {}

            try:
                response_json, wall_s = request_prediction(
                    session=session,
                    base_url=args.base_url,
                    model=args.model,
                    image_path=image_path,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    seed=args.seed,
                )

                raw_text = extract_text(response_json)
                predicted_label = normalize_prediction(raw_text)
                timings = response_json.get("timings") or {}

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"

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
            csv_file.flush()
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
    print(f"Images:       {len(results)}")
    print(f"Correct:      {correct_count}")
    print(f"Invalid:      {invalid_count}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")

    if latencies:
        print(f"Average wall: {statistics.mean(latencies):.3f} s")
        print(f"Median wall:  {statistics.median(latencies):.3f} s")
        print(f"P95 wall:     {percentile_95(latencies):.3f} s")

    print(f"CSV:          {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
