import numpy as np
from numpy.typing import NDArray

from forecasting.models.protocol import ForecastModel


class FakeModel:
    def __init__(self) -> None:
        self.last_value: float | None = None

    def fit(self, train: NDArray[np.float64]) -> None:
        self.last_value = float(train[-1])

    def predict(self, horizon: int) -> NDArray[np.float64]:
        if self.last_value is None:
            raise ValueError("The model must be fitted before prediction")

        return np.full(horizon, self.last_value, dtype=np.float64)


def test_fake_model_respects_forecast_interface() -> None:
    model: ForecastModel = FakeModel()
    train = np.array([10.0, 12.0, 14.0, 16.0], dtype=np.float64)

    model.fit(train)
    predictions = model.predict(2)

    np.testing.assert_array_equal(predictions, np.array([16.0, 16.0]))
