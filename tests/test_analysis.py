from src.analysis.compile_final_results import (
    GESTURE_FAMILY_MAP,
    LABELS,
    calculate_macro_metrics,
    gesture_family,
    percentile,
)


def test_gesture_family_rule():
    assert gesture_family("peace") == gesture_family("peace_inverted")
    assert gesture_family("stop") == gesture_family("stop_inverted")
    assert gesture_family("two_up") == gesture_family("two_up_inverted")
    assert gesture_family("three") == gesture_family("three2")
    assert gesture_family("palm") != gesture_family("stop")


def test_only_intended_labels_are_grouped():
    assert set(GESTURE_FAMILY_MAP).issubset(set(LABELS))
    assert "palm" not in GESTURE_FAMILY_MAP


def test_macro_metrics_perfect_predictions():
    true = list(LABELS)
    precision, recall, f1 = calculate_macro_metrics(true, true)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_percentile():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
