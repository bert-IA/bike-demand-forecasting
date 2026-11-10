import warnings

import numpy as np
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore[import-untyped]


class SeasonalExponentialSmoothing:
    """Seasonal exponential smoothing for one time series."""

    def __init__(self, season_length: int) -> None:
        if season_length <= 1:
            raise ValueError(f"season_length must be greater than 1, got {season_length}")
        self.season_length = season_length
        self._result = None

    def fit(self, train: NDArray[np.float64]) -> None:
        "Fit the model using one series of training values"
        self._result = None

        if train.ndim != 1:
            raise ValueError(f"train must be 1-D, got ndim {train.ndim}")

        if len(train) == 0:
            raise ValueError("train must not be empty")

        if not np.isfinite(train).all():
            raise ValueError("train must not contain NaN or infinite values")

        if len(train) < 2 * self.season_length:
            raise ValueError(
                f"train must contain at least {2 * self.season_length} values, got {len(train)}"
            )

        model = ExponentialSmoothing(
            train,
            trend=None,
            seasonal="add",
            seasonal_periods=self.season_length,
            initialization_method="estimated",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self._result = model.fit()

    def predict(self, horizon: int) -> NDArray[np.float64]:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        if self._result is None:
            raise ValueError("model must be fitted before prediction")

        predictions = self._result.forecast(horizon)

        return np.asarray(predictions, dtype=np.float64)
