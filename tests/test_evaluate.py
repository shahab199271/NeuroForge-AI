import pytest

from neuroforge.evaluate import calculate_metrics


def test_metrics_account_for_invalid_generations():
    metrics = calculate_metrics(
        ["positive", "negative", "positive"],
        ["positive", "negative", None],
    )
    assert metrics["accuracy_all"] == pytest.approx(2 / 3)
    assert metrics["invalid_rate"] == pytest.approx(1 / 3)
    assert metrics["accuracy_valid_only"] == pytest.approx(1.0)


def test_all_invalid_has_no_conditional_f1():
    metrics = calculate_metrics(["positive", "negative"], [None, None])
    assert metrics["accuracy_all"] == 0
    assert metrics["macro_f1_valid_only"] is None


def test_reject_misaligned_predictions():
    with pytest.raises(ValueError):
        calculate_metrics(["positive"], [])
