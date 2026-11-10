from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal

from forecasting.evaluation.backtest import (
    attach_actuals_to_forecasts,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
    run_backtest,
)
from forecasting.models.last_value import LastValueNaive


def test_last_value_naive_repeats_last_training_value() -> None:
    model = LastValueNaive()
    train = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)

    model.fit(train)
    predictions = model.predict(horizon=5)

    assert_array_equal(predictions, [15.0, 15.0, 15.0, 15.0, 15.0])


@pytest.mark.parametrize(
    "train",
    [
        np.array([], dtype=np.float64),
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array([np.nan, 2.0], dtype=np.float64),
        np.array([1.0, np.inf], dtype=np.float64),
        np.array([1.0, -np.inf], dtype=np.float64),
    ],
)
def test_last_value_naive_rejects_invalid_train(train: np.ndarray) -> None:
    model = LastValueNaive()

    with pytest.raises(ValueError, match="train"):
        model.fit(train)


def test_last_value_naive_rejects_prediction_before_fit() -> None:
    model = LastValueNaive()

    with pytest.raises(ValueError, match="fitted"):
        model.predict(horizon=2)


@pytest.mark.parametrize("horizon", [0, -1])
def test_last_value_naive_rejects_invalid_horizon(horizon: int) -> None:
    model = LastValueNaive()
    model.fit(np.array([5.0], dtype=np.float64))

    with pytest.raises(ValueError, match="horizon"):
        model.predict(horizon)


def test_last_value_naive_accepts_zero_and_refit() -> None:
    model = LastValueNaive()

    model.fit(np.array([5.0], dtype=np.float64))
    model.fit(np.array([3.0, 0.0], dtype=np.float64))

    assert_array_equal(model.predict(horizon=2), [0.0, 0.0])


def test_last_value_naive_backtest_to_metrics() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 182,
            "timestamp": [start + timedelta(days=day) for day in range(182)],
            "y": [float(day + 1) for day in range(182)],
        }
    )

    reports, forecasts, references = run_backtest(
        data,
        [cutoff],
        LastValueNaive,
        horizon=2,
        season_length=7,
    )
    evaluation = attach_actuals_to_forecasts(forecasts, data)
    wape_scores = evaluate_wape_by_series(evaluation)
    scaled_scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert reports["eligible"].item() is True
    assert_array_equal(
        forecasts.sort("timestamp")["y_pred"].to_numpy(),
        [180.0, 180.0],
    )
    assert wape_scores["n_evaluable"].item() == 2
    assert wape_scores["wape"].item() == pytest.approx(3 / 363)
    assert wape_scores["wape_status"].item() == "ok"

    assert scaled_scores["mase"].item() == pytest.approx(1.5 / 7)
    assert scaled_scores["rmsse"].item() == pytest.approx(np.sqrt(2.5 / 49))
    assert scaled_scores["mase_status"].item() == "ok"
    assert scaled_scores["rmsse_status"].item() == "ok"
