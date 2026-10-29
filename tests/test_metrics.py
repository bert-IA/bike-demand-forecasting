"""Tests for forecast accuracy metrics."""

from typing import Any

import pytest

from forecasting.evaluation.metrics import wape


def test_wape_perfect_forecast_is_zero() -> None:
    y_true = [10, 20]
    y_pred = [10, 20]
    assert wape(y_true, y_pred) == pytest.approx(0.0)


def test_wape_hand_computed() -> None:
    y_true = [10, 20]
    y_pred = [12, 18]
    assert wape(y_true, y_pred) == pytest.approx(4 / 30)


def test_wape_zero_actuals_raises() -> None:
    y_true = [0, 0, 0]
    y_pred = [2, 0, 1]
    with pytest.raises(ValueError, match="zero"):
        wape(y_true, y_pred)


def test_wape_length_mismatch_raises() -> None:
    y_true = [10, 20]
    y_pred = [12]
    with pytest.raises(ValueError, match="shape"):
        wape(y_true, y_pred)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([], []),  # empty
        ([1.0, float("nan")], [1, 2]),  # NaN
        ([[1, 2]], [[1, 2]]),  # 2-D
    ],
)
def test_wape_invalid_inputs_raise(y_true: list[Any], y_pred: list[Any]) -> None:
    with pytest.raises(ValueError):
        wape(y_true, y_pred)
