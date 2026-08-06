from src.evaluation.eval_qwen3vl_server import (
    LABELS,
    calculate_macro_f1,
    normalize_prediction,
    percentile,
)


def test_label_count_and_uniqueness():
    assert len(LABELS) == 19
    assert len(set(LABELS)) == 19


def test_normalize_prediction_exact_and_formatted():
    assert normalize_prediction("peace_inverted") == "peace_inverted"
    assert normalize_prediction("`stop`") == "stop"
    assert normalize_prediction("The label is two_up.") == "two_up"


def test_normalize_prediction_rejects_ambiguous_or_unknown():
    assert normalize_prediction("peace or stop") == "INVALID"
    assert normalize_prediction("unknown") == "INVALID"


def test_macro_f1_perfect_predictions():
    true = list(LABELS)
    assert calculate_macro_f1(true, true) == 1.0


def test_percentile_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
