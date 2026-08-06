import csv
import json
from collections import Counter
from pathlib import Path

import pytest

MODELS = ("q8", "q6", "q5", "q4", "q3")


def result_path(model: str) -> Path:
    return Path("results/image_results") / f"qwen3vl4b_{model}_336px_380_fullgpu.csv"


def test_final_prediction_csvs_are_paired_and_portable():
    paths = [result_path(model) for model in MODELS]
    if not all(path.is_file() for path in paths):
        pytest.skip("Final prediction CSVs are not present")

    reference_images = None
    for path in paths:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert len(rows) == 380
        images = [row["image_path"] for row in rows]
        assert len(set(images)) == 380
        assert all(not Path(image).is_absolute() for image in images)
        if reference_images is None:
            reference_images = images
        else:
            assert images == reference_images


def test_final_metadata_is_balanced():
    path = Path("data/hagrid_380_resize336/metadata.jsonl")
    if not path.is_file():
        pytest.skip("Final metadata is not present")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 380
    assert len({row["image_path"] for row in rows}) == 380
    assert set(Counter(row["label"] for row in rows).values()) == {20}
