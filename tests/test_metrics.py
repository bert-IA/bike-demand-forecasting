"""Tests for forecast accuracy metrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from forecasting.evaluation.metrics import (
    interval_coverage,
    mase,
    mean_interval_width,
    pinball_loss,
    rmsse,
    wape,
)

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
    with pytest.raises(ValueError, match="must have the same shape"):
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
    with pytest.raises(ValueError, match="must have the same shape"):
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
    with pytest.raises(ValueError, match="must have the same shape"):
        pinball_loss([10, 20], [12], quantile=0.9)


# interval coverage


@pytest.mark.parametrize(
    ("y_true", "lower", "upper", "expected"),
    [
        pytest.param([1, 2, 2, 4], [0, 0, 0, 0], [3, 3, 3, 3], 0.75, id="one-outside"),
        pytest.param([0, 3], [0, 0], [3, 3], 1.0, id="bounds-inclusive"),
        pytest.param([2, 3, 4, 2], [1, 1, 1, 1], [4, 5, 5, 5], 1.0, id="full-coverage"),
        pytest.param([2, 3, 4, 2], [3, 4, 5, 3], [4, 6, 6, 6], 0.0, id="no-coverage"),
        pytest.param([2], [2], [2], 1.0, id="zero-width-interval-is-valid"),
    ],
)
def test_interval_coverage_hand_computed(
    y_true: list[int], lower: list[int], upper: list[int], expected: float
) -> None:
    assert interval_coverage(y_true, lower, upper) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("y_true", "lower", "upper"),
    [
        pytest.param([1, 2, 2], [0, 0, 0], [3, 3, 3, 3], id="upper-longer"),
        pytest.param([1, 2, 2], [0, 0, 0], [3, 3], id="upper-shorter"),
        pytest.param([2, 2], [1, 1, 1], [5, 5], id="lower-longer"),
        pytest.param([2, 2], [1], [6, 6], id="lower-shorter"),
        pytest.param([1, 2, 2], [0, 0], [3, 3], id="y-true-longer"),
        pytest.param([1, 2], [0, 0, 0], [3, 3, 3], id="y-true-shorter"),
        pytest.param([1], [0, 0], [3, 3], id="y-true-size-one"),
    ],
)
def test_interval_coverage_length_mismatch_raises(
    y_true: list[int], lower: list[int], upper: list[int]
) -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        interval_coverage(y_true, lower, upper)


def test_interval_coverage_lower_above_upper_raises() -> None:
    with pytest.raises(ValueError, match="lower"):
        interval_coverage([3, 3], [2, 5], [4, 4])


@pytest.mark.parametrize(
    ("y_true", "lower", "upper", "message"),
    [
        pytest.param([float("nan"), 2], [0, 0], [3, 3], "y_true", id="nan-in-y_true"),
        pytest.param([1, 2], [float("nan"), 0], [3, 3], "lower", id="nan-in-lower"),
        pytest.param([1, 2], [0, 0], [float("nan"), 3], "upper", id="nan-in-upper"),
    ],
)
def test_interval_coverage_invalid_input_raises(
    y_true: list[Any], lower: list[Any], upper: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        interval_coverage(y_true, lower, upper)


# mean_interval_width


# --- mean_interval_width ---


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        pytest.param([1, 2, 2], [2, 4, 4], 5 / 3, id="hand-computed"),
        pytest.param([0, 2], [4, 3], 2.5, id="mixed-widths"),
        pytest.param([2, 2], [2, 2], 0.0, id="zero-width"),
    ],
)
def test_mean_interval_width_hand_computed(
    lower: list[int], upper: list[int], expected: float
) -> None:
    assert mean_interval_width(lower, upper) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        pytest.param([1, 2, 2], [0, 1, 3], id="one-position-inverted"),
        pytest.param([2, 2, 2], [1, 1, 1], id="all-inverted"),
        pytest.param([2], [1], id="single-point"),
    ],
)
def test_mean_interval_width_lower_above_upper_raises(lower: list[int], upper: list[int]) -> None:
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        mean_interval_width(lower, upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        pytest.param([1, 2], [3], id="upper-size-one"),
        pytest.param([1], [3, 4], id="lower-size-one"),
        pytest.param([1, 2, 3], [4, 5], id="upper-shorter"),
        pytest.param([1, 2], [4, 5, 6], id="upper-longer"),
    ],
)
def test_mean_interval_width_length_mismatch_raises(lower: list[int], upper: list[int]) -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        mean_interval_width(lower, upper)


@pytest.mark.parametrize(("lower", "upper", "message"), INVALID_SERIES)
def test_mean_interval_width_invalid_inputs_raise(
    lower: list[Any], upper: list[Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mean_interval_width(lower, upper)
