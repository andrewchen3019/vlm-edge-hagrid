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
            raise ValueError(
                f"Not enough images for label {label}: have {len(rows)}, need {per_class}"
            )
        rows = list(rows)
        rng.shuffle(rows)
        selected.extend(rows[:per_class])

    rng.shuffle(selected)
    return selected


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("-", "_")
    text = text.replace("no gesture", "no_gesture")
    text = text.replace("peace inverted", "peace_inverted")
    text = text.replace("stop inverted", "stop_inverted")
    text = text.replace("two up inverted", "two_up_inverted")
    text = text.replace("two up", "two_up")
    text = text.replace("thumbs up", "like")
    text = text.replace("thumb up", "like")
    text = text.replace("thumbs down", "dislike")
    text = text.replace("thumb down", "dislike")
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_generated_text(full_output: str) -> str:
    text = full_output.strip()

    # Remove ANSI color codes.
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    # Qwen/Jinja chat template: take final assistant section.
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # LLaVA/Vicuna fallback.
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]

    lines = []
    bad = [
        "llama_",
        "ggml_",
        "mtmd_",
        "common_",
        "load_",
        "main:",
        "system_info",
        "sampling:",
        "generate:",
        "perf",
        "clip",
        "cuda",
        "warn",
        "error:",
        "encoding mtmd",
        "chat template example",
        "llama_context",
        "llama_model",
        "llama_print",
    ]

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        lower = s.lower()

        # Skip timestamped llama.cpp logs like: 0.12.139.315 I ...
        if re.match(r"^\d+\.\d+\.\d+\.\d+\s+[iwe]\s+", lower):
            continue

        if any(b in lower for b in bad):
            continue

        s = s.replace("<|im_end|>", "").strip()
        s = s.replace("<|endoftext|>", "").strip()

        if s:
            lines.append(s)

    if lines:
        return lines[-1].strip()

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
    text = re.sub(r"\x1b\[[0-9;]*m", "", full_output)

    timings = {
        "mtmd_encode_ms": None,
        "prompt_eval_ms": None,
        "prompt_eval_tokens": None,
        "decode_eval_ms": None,
        "decode_eval_tokens": None,
        "total_ms": None,
        "total_tokens": None,
    }

    # Your llama-mtmd-cli usually prints this even when prompt/decode timing is unavailable.
    mtmd_pat = re.search(
        r"mtmd batch encoding done in\s+(\d+)\s*ms",
        text,
        re.IGNORECASE,
    )
    if mtmd_pat:
        timings["mtmd_encode_ms"] = float(mtmd_pat.group(1))

    prompt_pat = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)",
        text,
        re.IGNORECASE,
    )
    if prompt_pat:
        timings["prompt_eval_ms"] = float(prompt_pat.group(1))
        timings["prompt_eval_tokens"] = int(prompt_pat.group(2))

    decode_matches = re.findall(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)",
        text,
        re.IGNORECASE,
    )
    if decode_matches:
        timings["decode_eval_ms"] = float(decode_matches[-1][0])
        timings["decode_eval_tokens"] = int(decode_matches[-1][1])

    total_pat = re.search(
        r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)",
        text,
        re.IGNORECASE,
    )
    if total_pat:
        timings["total_ms"] = float(total_pat.group(1))
        timings["total_tokens"] = int(total_pat.group(2))

    return timings


def run_cli(row, args, raw_dir: Path, index: int):
    image_path = row.get("image_path") or row.get("crop_path") or row.get("full_path")
    true_label = row["label"]

    if not image_path:
        raise ValueError(f"No image path found in row: {row}")

    cmd = [
        args.cli,
        "--model", args.model,
        "--mmproj", args.mmproj,
        "--image", image_path,
        "--jinja",
        "-p", PROMPT,
        "--temp", str(args.temp),
        "-n", str(args.max_tokens),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "--fit", "off",
        "--no-mmproj-offload",
    ]

    if args.perf:
        cmd.append("--perf")

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
        "original_image_path": row.get("original_image_path", ""),
        "input_type": row.get("input_type", "full_image"),
        "true": true_label,
        "pred": pred,
        "correct": pred == true_label,
        "invalid": pred == "INVALID",
        "generated_text": generated,
        "full_output": full_output if args.save_full_output else "",
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
    parser.add_argument(
        "--model",
        default="models/qwen3-vl-4b-instruct-gguf/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--mmproj",
        default="models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
    )

    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--ngl", type=int, default=10)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--save-full-output", action="store_true")

    args = parser.parse_args()

    global PROMPT
    if args.prompt_file:
        PROMPT = Path(args.prompt_file).read_text(encoding="utf-8")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("results/raw_cli") / out_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = load_balanced_rows(args.metadata, args.per_class, args.seed)

    print("Balanced Qwen3-VL CLI eval")
    print("Images:", len(rows))
    print("Per class:", args.per_class)
    print("Model:", args.model)
    print("MMProj:", args.mmproj)
    print("Metadata:", args.metadata)
    print("Output CSV:", out_path)
    print("Raw logs:", raw_dir)

    results = []

    for i, row in enumerate(rows):
        image_path = row.get("image_path") or row.get("crop_path") or row.get("full_path")

        print("=" * 80)
        print(f"[{i + 1}/{len(rows)}]")
        print("true:", row["label"])
        print("image:", image_path)

        result = run_cli(row, args, raw_dir, i)

        print("generated:", repr(result["generated_text"]))
        print("pred:", result["pred"])
        print("correct:", result["correct"])
        print("wall_s:", f"{result['wall_s']:.2f}")
        print("mtmd_encode_ms:", result["mtmd_encode_ms"])
        print("returncode:", result["returncode"])

        results.append(result)
        pd.DataFrame(results).to_csv(out_path, index=False)

    df = pd.DataFrame(results)

    print("\nDONE")
    print("Saved:", out_path)
    print("Accuracy:", df["correct"].mean())
    print("Invalid:", int(df["invalid"].sum()), "/", len(df))
    print("Avg wall_s:", df["wall_s"].mean())

    if "mtmd_encode_ms" in df.columns:
        vals = pd.to_numeric(df["mtmd_encode_ms"], errors="coerce").dropna()
        if len(vals):
            print("Avg mtmd_encode_ms:", vals.mean())


if __name__ == "__main__":
    main()
