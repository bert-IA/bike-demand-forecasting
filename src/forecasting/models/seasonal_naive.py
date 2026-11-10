import numpy as np
from numpy.typing import NDArray


class SeasonalNaive:
    """Forecast by repeating the last observed seasonal cycle."""

    def __init__(self, season_length: int) -> None:
        if season_length < 1:
            raise ValueError(f"season_length must be at least 1, got {season_length}")
        self.season_length = season_length
        self._last_cycle: NDArray[np.float64] | None = None

    def fit(self, train: NDArray[np.float64]) -> None:
        if train.ndim != 1:
            raise ValueError(f"train must be 1-D, got ndim {train.ndim}")

        if len(train) < self.season_length:
            raise ValueError(
                f"train must contain at least {self.season_length} values, got {len(train)}"
            )

        if not np.isfinite(train).all():
            raise ValueError("train must not contain NaN or infinite values")
        self._last_cycle = train[-self.season_length :].copy()

    def predict(self, horizon: int) -> NDArray[np.float64]:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if self._last_cycle is None:
            raise ValueError("model must be fitted before prediction")

        return np.resize(self._last_cycle, horizon)
