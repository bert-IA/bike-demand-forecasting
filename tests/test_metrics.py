"""Tests for forecast accuracy metrics."""

from typing import Any

import pytest

from forecasting.evaluation.metrics import mase, wape


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


def test_mase_hand_computed() -> None:
    # Naive MAE on the train set is 1, model MAE is 2, so MASE is 2.
    y_train = [1, 2, 3, 4]
    y_true = [5]
    y_pred = [7]
    assert mase(y_true, y_pred, y_train, season_length=1) == pytest.approx(2.0)


def test_mase_non_constant_naive_errors() -> None:
    # Naive MAE is (1 + 2 + 4) / 3 = 7/3, model MAE is 2, so MASE is 6/7.
    y_train = [1, 2, 4, 8]
    y_true = [10, 12]
    y_pred = [9, 15]
    assert mase(y_true, y_pred, y_train, season_length=1) == pytest.approx(6 / 7)


def test_mase_constant_train_raises() -> None:
    # A constant train gives a naive MAE of 0, so MASE is undefined.
    y_train = [3, 3, 3, 3]
    y_true = [3, 4]
    y_pred = [5, 6]
    with pytest.raises(ValueError, match="naive MAE is zero"):
        mase(y_true, y_pred, y_train, season_length=1)


def test_mase_perfectly_seasonal_train_raises() -> None:
    # Each value equals the one season_length steps before, so naive MAE is 0.
    y_train = [1, 2, 1, 2]
    y_true = [1, 2]
    y_pred = [2, 1]
    with pytest.raises(ValueError, match="naive MAE is zero"):
        mase(y_true, y_pred, y_train, season_length=2)


def test_mase_train_too_short_raises() -> None:
    # With 2 values and season_length=2, no seasonal difference can be computed.
    y_train = [1, 2]
    y_true = [3, 4]
    y_pred = [2, 5]
    with pytest.raises(ValueError, match="more than season_length"):
        mase(y_true, y_pred, y_train, season_length=2)


def test_mase_invalid_season_length_raises() -> None:
    y_train = [1, 2, 4, 8]
    y_true = [10, 12]
    y_pred = [9, 15]
    with pytest.raises(ValueError, match="season_length must be at least 1"):
        mase(y_true, y_pred, y_train, season_length=0)


@pytest.mark.parametrize(
    "y_train",
    [[], [1.0, float("nan"), 3.0], [[1, 2, 3]]],
)
def test_mase_invalid_train_raises(y_train: list[Any]) -> None:
    with pytest.raises(ValueError, match="y_train"):
        mase([1, 2], [1, 2], y_train, season_length=1)
