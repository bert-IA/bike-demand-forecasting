"""Scale-free forecast accuracy metrics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_series(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert one input to a 1-D float array and validate it."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim {array.ndim}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return array


def _validate_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert actuals and forecasts to validated float arrays of the same shape."""
    actual = _validate_series(y_true, "y_true")
    forecast = _validate_series(y_pred, "y_pred")
    if actual.shape != forecast.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {actual.shape} and {forecast.shape}"
        )
    return actual, forecast


def _seasonal_naive_errors(y_train: ArrayLike, season_length: int) -> NDArray[np.float64]:
    """Return the seasonal differences of the training series."""
    if season_length < 1:
        raise ValueError(f"season_length must be at least 1, got {season_length}")
    train = _validate_series(y_train, "y_train")
    if train.size <= season_length:
        raise ValueError(
            "y_train needs more than season_length values, "
            f"got {train.size} with season_length={season_length}"
        )
    return train[season_length:] - train[:-season_length]


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum|y|."""
    actual, forecast = _validate_pair(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ValueError("sum of absolute actuals is zero, WAPE is undefined")
    return float(np.abs(actual - forecast).sum() / denominator)


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    season_length: int = 7,
) -> float:
    """Mean absolute scaled error against a seasonal naive on the training set."""
    actual, forecast = _validate_pair(y_true, y_pred)
    naive_errors = _seasonal_naive_errors(y_train, season_length)
    naive_mae = np.abs(naive_errors).mean()
    if naive_mae == 0:
        raise ValueError("naive MAE is zero on y_train, MASE is undefined")
    return float(np.abs(actual - forecast).mean() / naive_mae)


def rmsse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    season_length: int = 7,
) -> float:
    """Mean cube scaled error against a seasonal naive on the training set."""
    actual, forecast = _validate_pair(y_true, y_pred)
    naive_errors = _seasonal_naive_errors(y_train, season_length)
    naive_mse = np.square(naive_errors).mean()
    model_mse = np.square(actual - forecast).mean()
    if naive_mse == 0:
        raise ValueError("naive MSE is zero on y_train, RMSSE is undefined")
    return float(np.sqrt(model_mse / naive_mse))
