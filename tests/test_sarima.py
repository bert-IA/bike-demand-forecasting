import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]

from forecasting.evaluation.backtest import run_cutoff
from forecasting.models.sarima import SeasonalARIMA


@pytest.fixture
def synthetic_train() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    days = np.arange(210)

    return 100 + 10 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 5, 210)


def test_sarima_matches_statsmodels(
    synthetic_train: NDArray[np.float64],
) -> None:
    model = SeasonalARIMA()
    model.fit(synthetic_train)
    predictions = model.predict(14)

    expected = (
        SARIMAX(
            synthetic_train,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 7),
            trend="c",
        )
        .fit(disp=False, maxiter=200)
        .forecast(steps=14)
    )

    assert predictions.shape == (14,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, expected)


def test_predict_before_fit_is_rejected() -> None:
    with pytest.raises(ValueError, match="fitted"):
        SeasonalARIMA().predict(14)


@pytest.mark.parametrize("horizon", [0, -1])
def test_predict_rejects_invalid_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        SeasonalARIMA().predict(horizon)


def test_fit_rejects_multidimensional_data() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="1-D"):
        SeasonalARIMA().fit(train)


def test_fit_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        SeasonalARIMA().fit(np.array([], dtype=np.float64))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_data(value: float) -> None:
    train = np.array([1.0, value], dtype=np.float64)

    with pytest.raises(ValueError, match="NaN or infinite"):
        SeasonalARIMA().fit(train)


def test_failed_refit_invalidates_previous_result(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_train: NDArray[np.float64],
) -> None:
    model = SeasonalARIMA()
    model.fit(synthetic_train)

    def fake_fit(self: object, **kwargs: object) -> None:
        warnings.warn("Optimization failed", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(SARIMAX, "fit", fake_fit)

    with pytest.raises(ConvergenceWarning):
        model.fit(synthetic_train)

    with pytest.raises(ValueError, match="fitted"):
        model.predict(14)


def test_sarima_runs_in_backtest(
    synthetic_train: NDArray[np.float64],
) -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=209)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 210,
            "timestamp": [start + timedelta(days=day) for day in range(210)],
            "y": synthetic_train,
        }
    )

    reports, forecasts, references = run_cutoff(
        data,
        cutoff,
        SeasonalARIMA,
        horizon=14,
    )

    assert reports["eligible"].item() is True
    assert forecasts.height == 14
    assert forecasts["timestamp"].min() == cutoff + timedelta(days=1)
    assert forecasts["timestamp"].max() == cutoff + timedelta(days=14)
    assert np.isfinite(forecasts["y_pred"].to_numpy()).all()
    assert references.height == 1


def test_sarima_with_weekly_difference_and_seasonal_ma(
    synthetic_train: NDArray[np.float64],
) -> None:
    model = SeasonalARIMA(seasonal_order=(1, 1, 1, 7))
    model.fit(synthetic_train)
    predictions = model.predict(14)

    expected = (
        SARIMAX(
            synthetic_train,
            order=(1, 0, 0),
            seasonal_order=(1, 1, 1, 7),
            trend="c",
        )
        .fit(disp=False, maxiter=200)
        .forecast(steps=14)
    )

    assert predictions.shape == (14,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, expected)


def test_normalized_sarima_returns_predictions_in_original_units(
    synthetic_train: NDArray[np.float64],
) -> None:
    scale = np.abs(synthetic_train).max()

    model = SeasonalARIMA(
        seasonal_order=(1, 1, 1, 7),
        normalize=True,
    )
    model.fit(synthetic_train)
    predictions = model.predict(14)

    expected = (
        SARIMAX(
            synthetic_train / scale,
            order=(1, 0, 0),
            seasonal_order=(1, 1, 1, 7),
            trend="c",
        )
        .fit(disp=False, maxiter=200)
        .forecast(steps=14)
        * scale
    )

    np.testing.assert_allclose(predictions, expected)


def test_normalized_sarima_recalculates_scale_on_refit(
    synthetic_train: NDArray[np.float64],
) -> None:
    model = SeasonalARIMA(
        seasonal_order=(1, 1, 1, 7),
        normalize=True,
    )

    model.fit(synthetic_train)
    first_predictions = model.predict(14)

    model.fit(synthetic_train * 2)
    second_predictions = model.predict(14)

    np.testing.assert_allclose(second_predictions, first_predictions * 2)


def test_normalized_sarima_uses_neutral_scale_for_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def forecast(self, steps: int) -> NDArray[np.float64]:
            return np.full(steps, 2.0)

    class FakeSARIMAX:
        def __init__(self, train: NDArray[np.float64], **kwargs: object) -> None:
            np.testing.assert_array_equal(train, np.zeros(210))

        def fit(self, **kwargs: object) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr("forecasting.models.sarima.SARIMAX", FakeSARIMAX)

    model = SeasonalARIMA(normalize=True)
    model.fit(np.zeros(210))
    np.testing.assert_array_equal(model.predict(3), np.full(3, 2.0))


def test_normalized_sarima_with_powell(
    synthetic_train: NDArray[np.float64],
) -> None:
    scale = np.abs(synthetic_train).max()

    model = SeasonalARIMA(
        seasonal_order=(1, 1, 1, 7),
        normalize=True,
        optimizer="powell",
    )
    model.fit(synthetic_train)

    expected = (
        SARIMAX(
            synthetic_train / scale,
            order=(1, 0, 0),
            seasonal_order=(1, 1, 1, 7),
            trend="c",
        )
        .fit(disp=False, maxiter=200, method="powell")
        .forecast(steps=14)
        * scale
    )

    np.testing.assert_allclose(model.predict(14), expected)


def test_sarima_passes_configured_parameters_to_statsmodels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSARIMAX:
        def __init__(self, train: NDArray[np.float64], **kwargs: object) -> None:
            captured["model"] = kwargs

        def fit(self, **kwargs: object) -> object:
            captured["fit"] = kwargs
            return object()

    monkeypatch.setattr("forecasting.models.sarima.SARIMAX", FakeSARIMAX)

    model = SeasonalARIMA(
        order=(0, 0, 1),
        seasonal_order=(1, 1, 1, 7),
        trend="n",
        optimizer="powell",
        maxiter=50,
    )
    model.fit(np.array([1.0, 2.0, 3.0]))

    assert captured["model"] == {
        "order": (0, 0, 1),
        "seasonal_order": (1, 1, 1, 7),
        "trend": "n",
    }
    assert captured["fit"] == {
        "disp": False,
        "maxiter": 50,
        "method": "powell",
    }
