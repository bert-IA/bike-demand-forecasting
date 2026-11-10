from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from forecasting.evaluation.backtest import (
    attach_actuals_to_forecasts,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
    run_backtest,
    run_cutoff,
)
from forecasting.models.seasonal_naive import SeasonalNaive


def test_seasonal_naive_repeats_last_cycle() -> None:
    train = np.array([99.0, 10.0, 12.0, 8.0], dtype=np.float64)
    model = SeasonalNaive(season_length=3)

    model.fit(train)
    predictions = model.predict(horizon=5)

    assert_array_equal(predictions, [10.0, 12.0, 8.0, 10.0, 12.0])


@pytest.mark.parametrize("season_length", [0, -1])
def test_seasonal_naive_rejects_invalid_season_length(season_length: int) -> None:
    with pytest.raises(ValueError, match="season_length"):
        SeasonalNaive(season_length=season_length)


def test_seasonal_naive_rejects_short_train() -> None:
    model = SeasonalNaive(season_length=3)
    train = np.array([10.0, 12.0], dtype=np.float64)

    with pytest.raises(ValueError, match="train"):
        model.fit(train)


def test_seasonal_naive_rejects_prediction_before_fit() -> None:
    model = SeasonalNaive(season_length=3)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(horizon=2)


def test_seasonal_naive_accepts_exactly_one_cycle() -> None:
    model = SeasonalNaive(season_length=3)
    model.fit(np.array([10.0, 12.0, 8.0], dtype=np.float64))

    assert_array_equal(model.predict(horizon=3), [10.0, 12.0, 8.0])


def test_seasonal_naive_copies_training_cycle() -> None:
    train = np.array([10.0, 12.0, 8.0], dtype=np.float64)
    model = SeasonalNaive(season_length=3)
    model.fit(train)

    train[-1] = 999.0

    assert_array_equal(model.predict(horizon=3), [10.0, 12.0, 8.0])


@pytest.mark.parametrize("horizon", [0, -1])
def test_seasonal_naive_rejects_invalid_horizon(horizon: int) -> None:
    model = SeasonalNaive(season_length=3)
    model.fit(np.array([10.0, 12.0, 8.0], dtype=np.float64))

    with pytest.raises(ValueError, match="horizon"):
        model.predict(horizon)


@pytest.mark.parametrize(
    "train",
    [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.array([1.0, np.nan, 3.0], dtype=np.float64),
        np.array([1.0, np.inf, 3.0], dtype=np.float64),
        np.array([1.0, -np.inf, 3.0], dtype=np.float64),
    ],
)
def test_seasonal_naive_rejects_invalid_train(train: NDArray[np.float64]) -> None:
    model = SeasonalNaive(season_length=3)

    with pytest.raises(ValueError, match="train"):
        model.fit(train)


def test_seasonal_naive_rejects_nonfinite_value_before_last_cycle() -> None:
    model = SeasonalNaive(season_length=3)
    train = np.array([np.nan, 10.0, 12.0, 8.0], dtype=np.float64)

    with pytest.raises(ValueError, match="train"):
        model.fit(train)


def test_seasonal_naive_refit_replaces_last_cycle() -> None:
    model = SeasonalNaive(season_length=2)

    model.fit(np.array([1.0, 2.0], dtype=np.float64))
    model.fit(np.array([7.0, 8.0], dtype=np.float64))

    assert_array_equal(model.predict(horizon=3), [7.0, 8.0, 7.0])


def test_seasonal_naive_with_run_cutoff() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [float(day + 1) for day in range(180)],
        }
    )

    def make_model() -> SeasonalNaive:
        return SeasonalNaive(season_length=7)

    report, forecasts, references = run_cutoff(
        data,
        cutoff,
        make_model,
        horizon=9,
        season_length=7,
    )

    assert report["eligible"].item() is True
    assert_array_equal(
        forecasts.sort("timestamp")["y_pred"].to_numpy(),
        [174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 174.0, 175.0],
    )
    assert references["naive_mae"].item() == pytest.approx(7.0)
    assert references["naive_mse"].item() == pytest.approx(49.0)


def test_seasonal_naive_run_cutoff_keeps_series_separate() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)
    values_a = [float(day + 1) for day in range(180)]
    values_b = [float(2 * (day + 1)) for day in range(180)]

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180 + ["B"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)] * 2,
            "y": values_a + values_b,
        }
    )

    def make_model() -> SeasonalNaive:
        return SeasonalNaive(season_length=3)

    report, forecasts, _ = run_cutoff(data, cutoff, make_model, horizon=4, season_length=3)

    assert report["eligible"].sum() == 2

    predictions_a = forecasts.filter(pl.col("series_id") == "A").sort("timestamp")
    predictions_b = forecasts.filter(pl.col("series_id") == "B").sort("timestamp")

    assert_array_equal(predictions_a["y_pred"].to_numpy(), [178, 179, 180, 178])
    assert_array_equal(predictions_b["y_pred"].to_numpy(), [356, 358, 360, 356])


def test_seasonal_naive_backtest_to_metrics() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 182,
            "timestamp": [start + timedelta(days=day) for day in range(182)],
            "y": [float(day + 1) for day in range(182)],
        }
    )

    def make_model() -> SeasonalNaive:
        return SeasonalNaive(season_length=7)

    report, forecasts, references = run_backtest(
        data,
        [cutoff],
        make_model,
        horizon=2,
        season_length=7,
    )
    evaluation = attach_actuals_to_forecasts(forecasts, data)
    wape_scores = evaluate_wape_by_series(evaluation)
    scaled_scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert report["eligible"].item() is True
    assert_array_equal(
        forecasts.sort("timestamp")["y_pred"].to_numpy(),
        [174.0, 175.0],
    )

    assert wape_scores["n_evaluable"].item() == 2
    assert wape_scores["wape"].item() == pytest.approx(14 / 363)
    assert wape_scores["wape_status"].item() == "ok"

    assert scaled_scores["n_evaluable"].item() == 2
    assert scaled_scores["mase"].item() == pytest.approx(1.0)
    assert scaled_scores["rmsse"].item() == pytest.approx(1.0)
    assert scaled_scores["mase_status"].item() == "ok"
    assert scaled_scores["rmsse_status"].item() == "ok"
