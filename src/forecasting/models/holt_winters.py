from forecasting.models.exponential_smoothing import SeasonalExponentialSmoothing


class DampedHoltWinters(SeasonalExponentialSmoothing):
    """Holt-Winters with additive seasonality and a damped additive trend."""

    def __init__(self, season_length: int) -> None:
        super().__init__(season_length, with_damped_trend=True)
