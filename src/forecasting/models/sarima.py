import warnings

import numpy as np
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]


class SeasonalARIMA:
    """SARIMA with a weekly seasonal cycle."""

    def __init__(
        self,
        *,
        seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 7),
        normalize: bool = False,
        optimizer: str = "lbfgs",
        order: tuple[int, int, int] = (1, 0, 0),
        trend: str = "c",
        maxiter: int = 200,
    ) -> None:
        self.seasonal_order = seasonal_order
        self.normalize = normalize
        self.optimizer = optimizer
        self._scale = 1.0
        self._result = None
        self.order = order
        self.trend = trend
        self.maxiter = maxiter

    def fit(self, train: NDArray[np.float64]) -> None:
        """Fit one training series."""
        self._result = None
        self._scale = 1.0

        if train.ndim != 1:
            raise ValueError(f"train must be 1-D, got ndim {train.ndim}")

        if len(train) == 0:
            raise ValueError("train must not be empty")

        if not np.isfinite(train).all():
            raise ValueError("train must not contain NaN or infinite values")

        scale = 1.0
        if self.normalize:
            maximum = float(np.abs(train).max())
            if maximum > 0:
                scale = maximum

        model = SARIMAX(
            train / scale,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self._result = model.fit(
                disp=False,
                maxiter=self.maxiter,
                method=self.optimizer,
            )
        self._scale = scale

    def predict(self, horizon: int) -> NDArray[np.float64]:
        """Forecast the requested number of days."""
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        if self._result is None:
            raise ValueError("model must be fitted before prediction")

        predictions = self._result.forecast(steps=horizon)
        return np.asarray(predictions, dtype=np.float64) * self._scale
