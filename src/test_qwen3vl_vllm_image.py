#!/usr/bin/env python3

import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


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


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="data/hagrid_day1/images/fist/fist_00002.jpg",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    client = OpenAI(
        base_url=args.base_url,
        api_key="not-needed",
    )

    prompt = (
        "Classify the hand gesture in this image. "
        f"Allowed labels: {', '.join(LABELS)}. "
        "Return only one exact label."
    )

    response = client.chat.completions.create(
        model="qwen3-vl-4b-awq",
        messages=[
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
        temperature=0,
        max_tokens=8,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
