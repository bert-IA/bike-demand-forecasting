"""Tests for forecast accuracy metrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from forecasting.evaluation.metrics import mase, pinball_loss, rmsse, wape

INVALID_SERIES = [
    ([], [], "empty"),
    ([1.0, float("nan")], [1, 2], "NaN"),
    ([1.0, float("inf")], [1, 2], "NaN or infinite"),
    ([[1, 2]], [[1, 2]], "1-D"),
]

INVALID_TRAIN = [
    ([], "y_train must not be empty"),
    ([1.0, float("nan"), 3.0], "y_train must not contain"),
    ([[1, 2, 3]], "y_train must be 1-D"),
]


# --- wape ---


def test_wape_perfect_forecast_is_zero() -> None:
    assert wape([10, 20], [10, 20]) == pytest.approx(0.0)


def test_wape_hand_computed() -> None:
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_wape_accepts_negative_actuals() -> None:
    # The denominator is sum|y| = 10, so a zero sum of signed values is not an error.
    assert wape([-5, 5], [-5, 5]) == pytest.approx(0.0)


def test_wape_penalises_negative_forecast() -> None:
    # |10 - (-5)| / |10| = 1.5
    assert wape([10], [-5]) == pytest.approx(1.5)


def test_wape_does_not_modify_inputs() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    wape(y_true, y_pred)
    assert y_true.tolist() == [10.0, 20.0]
    assert y_pred.tolist() == [12.0, 18.0]


def test_wape_zero_actuals_raises() -> None:
    with pytest.raises(ValueError, match="zero"):
        wape([0, 0, 0], [2, 0, 1])


def test_wape_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        wape([10, 20], [12])


@pytest.mark.parametrize(("y_true", "y_pred", "message"), INVALID_SERIES)
def test_wape_invalid_inputs_raise(y_true: list[Any], y_pred: list[Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wape(y_true, y_pred)


# --- mase ---


def test_mase_hand_computed() -> None:
    # Naive MAE on the train set is 1, model MAE is 2, so MASE is 2.
    assert mase([5], [7], [1, 2, 3, 4], season_length=1) == pytest.approx(2.0)


def test_mase_non_constant_naive_errors() -> None:
    # Naive MAE is (1 + 2 + 4) / 3 = 7/3, model MAE is 2, so MASE is 6/7.
    assert mase([10, 12], [9, 15], [1, 2, 4, 8], season_length=1) == pytest.approx(6 / 7)


def test_mase_constant_train_raises() -> None:
    # A constant train gives a naive MAE of 0, so MASE is undefined.
    with pytest.raises(ValueError, match="naive MAE is zero"):
        mase([3, 4], [5, 6], [3, 3, 3, 3], season_length=1)


def test_mase_perfectly_seasonal_train_raises() -> None:
    # Each value equals the one season_length steps before, so naive MAE is 0.
    with pytest.raises(ValueError, match="naive MAE is zero"):
        mase([1, 2], [2, 1], [1, 2, 1, 2], season_length=2)


# --- rmsse ---


def test_rmsse_hand_computed() -> None:
    # Naive MSE is (1 + 4 + 16) / 3 = 7, model MSE is (1 + 9) / 2 = 5.
    result = rmsse([10, 12], [9, 15], [1, 2, 4, 8], season_length=1)
    assert result == pytest.approx(np.sqrt(5 / 7))


def test_rmsse_constant_train_raises() -> None:
    with pytest.raises(ValueError, match="naive MSE is zero"):
        rmsse([3, 4], [5, 6], [3, 3, 3, 3], season_length=1)


def test_rmsse_perfectly_seasonal_train_raises() -> None:
    with pytest.raises(ValueError, match="naive MSE is zero"):
        rmsse([1, 2], [2, 1], [1, 2, 1, 2], season_length=2)


def test_rmsse_penalises_concentrated_errors_more_than_mase() -> None:
    # Both forecasts have a MAE of 2, but the second concentrates the error on one day.
    y_train = [1, 2, 4, 8]
    y_true = [10, 10]
    spread = [8, 12]
    concentrated = [10, 14]
    assert mase(y_true, concentrated, y_train, 1) == pytest.approx(mase(y_true, spread, y_train, 1))
    assert rmsse(y_true, concentrated, y_train, 1) > rmsse(y_true, spread, y_train, 1)


# --- shared checks for mase and rmsse ---


@pytest.mark.parametrize("metric", [mase, rmsse])
def test_scaled_metrics_invalid_season_length_raises(metric: Callable[..., float]) -> None:
    with pytest.raises(ValueError, match="season_length must be at least 1"):
        metric([10, 12], [9, 15], [1, 2, 4, 8], season_length=0)


@pytest.mark.parametrize("metric", [mase, rmsse])
def test_scaled_metrics_train_too_short_raises(metric: Callable[..., float]) -> None:
    # With 2 values and season_length=2, no seasonal difference can be computed.
    with pytest.raises(ValueError, match="more than season_length"):
        metric([3, 4], [2, 5], [1, 2], season_length=2)


@pytest.mark.parametrize("metric", [mase, rmsse])
def test_scaled_metrics_length_mismatch_raises(metric: Callable[..., float]) -> None:
    with pytest.raises(ValueError, match="shape"):
        metric([10, 20], [12], [1, 2, 4, 8], season_length=1)


@pytest.mark.parametrize("metric", [mase, rmsse])
@pytest.mark.parametrize(("y_true", "y_pred", "message"), INVALID_SERIES)
def test_scaled_metrics_invalid_inputs_raise(
    metric: Callable[..., float], y_true: list[Any], y_pred: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric(y_true, y_pred, [1, 2, 4, 8], season_length=1)


@pytest.mark.parametrize("metric", [mase, rmsse])
@pytest.mark.parametrize(("y_train", "message"), INVALID_TRAIN)
def test_scaled_metrics_invalid_train_raises(
    metric: Callable[..., float], y_train: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric([1, 2], [1, 2], y_train, season_length=1)


# pinball loss


def test_pinball_underforecast() -> None:
    # Réel 10, prévision 8 : erreur positive de 2.
    # Pour q=0.9, la perte vaut 0.9 * 2 = 1.8.
    assert pinball_loss([10], [8], quantile=0.9) == pytest.approx(1.8)


def test_pinball_overforecast() -> None:
    # Réel 10, prévision 12 : erreur négative de -2.
    # Pour q=0.9, la perte vaut (1 - 0.9) * 2 = 0.2.
    assert pinball_loss([10], [12], quantile=0.9) == pytest.approx(0.2)


def test_pinball_median_is_half_mae() -> None:
    y_true = [10, 20]
    y_pred = [12, 16]

    # MAE = (2 + 4) / 2 = 3 ; pinball à q=0.5 = MAE / 2.
    assert pinball_loss(y_true, y_pred, quantile=0.5) == pytest.approx(1.5)


@pytest.mark.parametrize("quantile", [0, 1, -0.1, 1.1])
def test_pinball_invalid_quantile_raises(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([10, 20], [12, 18], quantile=quantile)


def test_pinball_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        pinball_loss([10, 20], [12], quantile=0.9)
