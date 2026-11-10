import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore[import-untyped]

from forecasting.evaluation.backtest import (
    attach_actuals_to_forecasts,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
    run_backtest,
    run_cutoff,
)
from forecasting.models.exponential_smoothing import (
    SeasonalExponentialSmoothing,
)


def test_prediction_before_fit_is_rejected() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(horizon=14)


@pytest.mark.parametrize("season_length", [0, -1, 1])
def test_rejects_invalid_season_length(season_length: int) -> None:
    with pytest.raises(ValueError, match="season_length"):
        SeasonalExponentialSmoothing(season_length=season_length)


def test_fit_and_predict_repeats_weekly_profile() -> None:
    week = np.array([100, 120, 110, 115, 105, 80, 70], dtype=np.float64)
    train = np.tile(week, 8)

    model = SeasonalExponentialSmoothing(season_length=7)
    model.fit(train)

    predictions = model.predict(horizon=14)

    assert predictions.shape == (14,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions[:7], predictions[7:], atol=1e-8)


def test_fit_rejects_multidimensional_train() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="train must be 1-D"):
        model.fit(train)


def test_fit_rejects_empty_train() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="train must not be empty"):
        model.fit(train)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_values(invalid_value: float) -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.array([1.0, 2.0, invalid_value], dtype=np.float64)

    with pytest.raises(ValueError, match="train must not contain NaN or infinite values"):
        model.fit(train)


def test_fit_rejects_less_than_two_seasonal_cycles() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.arange(13, dtype=np.float64)

    with pytest.raises(ValueError, match="train must contain at least 14 values"):
        model.fit(train)


@pytest.mark.parametrize("horizon", [0, -1])
def test_predict_rejects_invalid_horizon(horizon: int) -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.arange(28, dtype=np.float64) + 10
    model.fit(train)

    with pytest.raises(ValueError, match="horizon"):
        model.predict(horizon)


def test_fit_accepts_two_complete_seasonal_cycles() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.arange(14, dtype=np.float64) + 10

    model.fit(train)
    predictions = model.predict(horizon=3)

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_failed_refit_invalidates_previous_result() -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    valid_train = np.arange(28, dtype=np.float64) + 10
    model.fit(valid_train)

    invalid_train = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="train must not be empty"):
        model.fit(invalid_train)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(horizon=7)


def test_nonconvergent_refit_invalidates_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SeasonalExponentialSmoothing(season_length=7)
    train = np.arange(28, dtype=np.float64) + 10
    model.fit(train)

    def fit_with_convergence_warning(self: object) -> None:
        warnings.warn("Optimization failed to converge", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(
        ExponentialSmoothing,
        "fit",
        fit_with_convergence_warning,
    )

    with pytest.raises(ConvergenceWarning):
        model.fit(train)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(horizon=7)


def test_exponential_smoothing_with_run_cutoff() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [float(day + 10) for day in range(180)],
        }
    )

    def make_model() -> SeasonalExponentialSmoothing:
        return SeasonalExponentialSmoothing(season_length=7)

    report, forecasts, references = run_cutoff(
        data,
        cutoff,
        make_model,
        horizon=14,
        season_length=7,
    )

    assert report["eligible"].item() is True
    assert forecasts.height == 14
    assert forecasts["timestamp"].min() == cutoff + timedelta(days=1)
    assert forecasts["timestamp"].max() == cutoff + timedelta(days=14)
    assert np.isfinite(forecasts["y_pred"].to_numpy()).all()
    assert references.height == 1


def test_exponential_smoothing_backtest_to_metrics() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 194,
            "timestamp": [start + timedelta(days=day) for day in range(194)],
            "y": [float(day + 10) for day in range(194)],
        }
    )

    def make_model() -> SeasonalExponentialSmoothing:
        return SeasonalExponentialSmoothing(season_length=7)

    report, forecasts, references = run_backtest(
        data,
        [cutoff],
        make_model,
        horizon=14,
        season_length=7,
    )
    evaluation = attach_actuals_to_forecasts(forecasts, data)
    wape_scores = evaluate_wape_by_series(evaluation)
    scaled_scores = evaluate_scaled_metrics_by_series(evaluation, references)

    actual = np.arange(190, 204, dtype=np.float64)
    predicted = forecasts.sort("timestamp")["y_pred"].to_numpy()
    errors = actual - predicted

    assert report["eligible"].item() is True
    assert wape_scores["n_evaluable"].item() == 14
    assert wape_scores["wape_status"].item() == "ok"
    assert wape_scores["wape"].item() == pytest.approx(np.abs(errors).sum() / actual.sum())

    assert scaled_scores["mase_status"].item() == "ok"
    assert scaled_scores["rmsse_status"].item() == "ok"
    assert scaled_scores["mase"].item() == pytest.approx(np.abs(errors).mean() / 7)
    assert scaled_scores["rmsse"].item() == pytest.approx(np.sqrt(np.square(errors).mean() / 49))


def test_run_cutoff_fits_each_series_separately() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    dates = [start + timedelta(days=day) for day in range(180)]
    values_a = [float(day + 10) for day in range(180)]
    values_b = [float(day + 1010) for day in range(180)]

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180 + ["B"] * 180,
            "timestamp": dates + dates,
            "y": values_a + values_b,
        }
    )

    def make_model() -> SeasonalExponentialSmoothing:
        return SeasonalExponentialSmoothing(season_length=7)

    report, forecasts, references = run_cutoff(
        data,
        cutoff,
        make_model,
        horizon=14,
        season_length=7,
    )

    forecasts_a = forecasts.filter(pl.col("series_id") == "A").sort("timestamp")
    forecasts_b = forecasts.filter(pl.col("series_id") == "B").sort("timestamp")

    assert report["eligible"].sum() == 2
    assert references.height == 2
    assert forecasts_a.height == 14
    assert forecasts_b.height == 14
    assert forecasts_a["timestamp"].to_list() == forecasts_b["timestamp"].to_list()

    predictions_a = forecasts_a["y_pred"].to_numpy()
    predictions_b = forecasts_b["y_pred"].to_numpy()

    assert np.isfinite(predictions_a).all()
    assert np.isfinite(predictions_b).all()

    # Les deux séries ont des niveaux très différents.
    assert predictions_b.mean() - predictions_a.mean() > 900


def test_backtest_propagates_convergence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [float(day + 10) for day in range(180)],
        }
    )

    def make_model() -> SeasonalExponentialSmoothing:
        return SeasonalExponentialSmoothing(season_length=7)

    def fit_with_convergence_warning(self: object) -> None:
        warnings.warn("Optimization failed to converge", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(
        ExponentialSmoothing,
        "fit",
        fit_with_convergence_warning,
    )

    with pytest.raises(ConvergenceWarning, match="failed to converge"):
        run_backtest(
            data,
            [cutoff],
            make_model,
            horizon=14,
            season_length=7,
        )
