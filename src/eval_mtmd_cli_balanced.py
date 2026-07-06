import argparse
import json
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


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


PROMPT = """There is a hand gesture in this image. Choose the exact HaGRID label that best describes it:

fist = closed fist, all fingers curled.
palm = open hand, five fingers extended, palm facing camera.
like = thumb up, other fingers curled.
dislike = thumb down, other fingers curled.
ok = thumb and index form a circle, other fingers extended.
peace = index and middle spread in a V, palm facing camera.
peace_inverted = peace sign with back of hand facing camera.
stop = vertical open palm, fingers upward, palm facing camera.
stop_inverted = stop gesture with back of hand facing camera.
rock = index and little finger extended; middle/ring folded.
call = thumb and little finger extended like phone gesture.
mute = one finger held near lips / silence gesture.
one = only index finger extended.
two_up = index and middle extended upward, close together, not V-shaped.
two_up_inverted = two_up with back of hand facing camera.
three = index, middle, ring extended.
three2 = thumb, index, middle extended.
four = four fingers extended, thumb folded.
no_gesture = no clear matching gesture.

Return only the exact label."""

def load_balanced_rows(metadata_path: str, per_class: int, seed: int):
    buckets = defaultdict(list)

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            buckets[row["label"]].append(row)

    rng = random.Random(seed)
    selected = []

    for label in LABELS:
        rows = buckets[label]
        if len(rows) < per_class:
            raise ValueError(f"Not enough images for label {label}: have {len(rows)}, need {per_class}")
        rng.shuffle(rows)
        selected.extend(rows[:per_class])

    rng.shuffle(selected)
    return selected


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", "_")
    text = text.replace("no gesture", "no_gesture")
    text = text.replace("peace inverted", "peace_inverted")
    text = text.replace("stop inverted", "stop_inverted")
    text = text.replace("two up inverted", "two_up_inverted")
    text = text.replace("two up", "two_up")
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_generated_text(full_output: str) -> str:
    """
    Try to isolate only the model's generated answer, not the prompt/logs.
    """
    text = full_output

    # If the prompt is echoed, take the last part after ASSISTANT:
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]

    lines = []
    bad = [
        "llama_",
        "ggml_",
        "mtmd_",
        "common_",
        "load_",
        "print_timing",
        "prompt eval time",
        "eval time",
        "total time",
        "main:",
        "system_info",
        "sampling:",
        "generate:",
        "perf",
        "clip",
        "cuda",
        "warn",
        "error:",
    ]

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        lower = s.lower()

        # Skip timestamped llama.cpp log lines
        if re.match(r"^\d+\.\d+\.\d+\.\d+\s+[iwe]\s+", lower):
            continue

        if any(b in lower for b in bad):
            continue

        lines.append(s)

    # Usually the answer is the first/last short non-log line.
    if lines:
        return "\n".join(lines).strip()

    return text.strip()


def parse_prediction(generated_text: str) -> str:
    cleaned = clean_text(generated_text)

    if cleaned in LABELS:
        return cleaned

    # Prefer longest labels first, e.g. stop_inverted before stop.
    for label in sorted(LABELS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(label) + r"\b", cleaned):
            return label

    return "INVALID"


def parse_timings(full_output: str):
    timings = {
        "prompt_eval_ms": None,
        "prompt_eval_tokens": None,
        "decode_eval_ms": None,
        "decode_eval_tokens": None,
        "total_ms": None,
        "total_tokens": None,
    }

    prompt_pat = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        full_output,
        re.IGNORECASE,
    )
    if prompt_pat:
        timings["prompt_eval_ms"] = float(prompt_pat.group(1))
        timings["prompt_eval_tokens"] = int(prompt_pat.group(2))

    # Match eval time but not prompt eval time
    decode_pat = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        full_output,
        re.IGNORECASE,
    )
    if decode_pat:
        timings["decode_eval_ms"] = float(decode_pat.group(1))
        timings["decode_eval_tokens"] = int(decode_pat.group(2))

    total_pat = re.search(
        r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        full_output,
        re.IGNORECASE,
    )
    if total_pat:
        timings["total_ms"] = float(total_pat.group(1))
        timings["total_tokens"] = int(total_pat.group(2))

    return timings


def run_cli(row, args, raw_dir: Path, index: int):
    image_path = row.get("image_path") or row.get("crop_path") or row.get("full_path")
    true_label = row["label"]

    cmd = [
        args.cli,
        "-m", args.model,
        "--mmproj", args.mmproj,
        "--chat-template", args.chat_template,
        "--image", image_path,
        "-p", PROMPT,
        "--temp", "0",
        "-n", str(args.max_tokens),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "--fit", "off",
        "--no-mmproj-offload",
        "--perf"
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    wall_s = time.perf_counter() - t0

    full_output = proc.stdout
    generated = extract_generated_text(full_output)
    pred = parse_prediction(generated)
    timings = parse_timings(full_output)

    raw_path = raw_dir / f"{index:04d}_{true_label}_{Path(image_path).stem}.txt"
    raw_path.write_text(full_output, encoding="utf-8")

    return {
        "id": row.get("id", ""),
        "image_path": image_path,
        "true": true_label,
        "pred": pred,
        "correct": pred == true_label,
        "invalid": pred == "INVALID",
        "generated_text": generated,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "raw_log_path": str(raw_path),
        **timings,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", default="data/hagrid_day1/metadata.jsonl")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)

    parser.add_argument("--cli", default="./llama.cpp/build/bin/llama-mtmd-cli")
    parser.add_argument("--model", default="models/llava-v1.5-7b-second-state/llava-v1.5-7b-Q4_K_M.gguf")
    parser.add_argument("--mmproj", default="models/llava-v1.5-7b-second-state/llava-v1.5-7b-mmproj-model-f16.gguf")
    parser.add_argument("--chat-template", default="vicuna")

    parser.add_argument("--ngl", type=int, default=10)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("results/raw_cli") / out_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = load_balanced_rows(args.metadata, args.per_class, args.seed)

    print(f"Balanced CLI eval")
    print(f"Images: {len(rows)}")
    print(f"Per class: {args.per_class}")
    print(f"Output CSV: {out_path}")
    print(f"Raw logs: {raw_dir}")

    results = []

    for i, row in enumerate(rows):
        print("=" * 80)
        print(f"[{i + 1}/{len(rows)}]")
        print("true:", row["label"])
        print("image:", row.get("image_path") or row.get("crop_path") or row.get("full_path"))

        result = run_cli(row, args, raw_dir, i)
        print("generated:", repr(result["generated_text"]))
        print("pred:", result["pred"])
        print("correct:", result["correct"])
        print("wall_s:", f"{result['wall_s']:.2f}")
        print("prompt_eval_ms:", result["prompt_eval_ms"])
        print("decode_eval_ms:", result["decode_eval_ms"])
        print("total_ms:", result["total_ms"])
        print("returncode:", result["returncode"])

        results.append(result)
        pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)

    print("\nDONE")
    print("Saved:", out_path)
    print("Accuracy:", df["correct"].mean())
    print("Invalid:", df["invalid"].sum(), "/", len(df))
    print("Avg wall_s:", df["wall_s"].mean())
    if "total_ms" in df.columns:
        print("Avg internal total_ms:", df["total_ms"].mean())


if __name__ == "__main__":
    main()
