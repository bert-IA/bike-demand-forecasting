import numpy as np
from numpy.typing import NDArray


class LastValueNaive:
    """Forecast by repeating the last training value."""

    def __init__(self) -> None:
        self._last_value: np.float64 | None = None

    def fit(self, train: NDArray[np.float64]) -> None:
        if train.ndim != 1:
            raise ValueError(f"train must be 1-D, got ndim {train.ndim}")

        if len(train) == 0:
            raise ValueError("train must not be empty")

        if not np.isfinite(train).all():
            raise ValueError("train must not contain NaN or infinite values")

        self._last_value = train[-1]

    def predict(self, horizon: int) -> NDArray[np.float64]:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")

        if self._last_value is None:
            raise ValueError("model must be fitted before prediction")

        return np.full(horizon, self._last_value, dtype=np.float64)
