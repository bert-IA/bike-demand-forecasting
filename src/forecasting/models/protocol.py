from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ForecastModel(Protocol):
    """Interface for a local point forecasting model."""

    def fit(self, train: NDArray[np.float64]) -> None:
        """Fit the model using one series of training values."""
        ...

    def predict(self, horizon: int) -> NDArray[np.float64]:
        """Return one forecast per day in the horizon."""
        ...
