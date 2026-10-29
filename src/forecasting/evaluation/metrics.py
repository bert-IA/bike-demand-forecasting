"""Scale-free forecast accuracy metrics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert inputs to 1-D float arrays and validate them."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got ndim {true.ndim} and {pred.ndim}")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError("y_true and y_pred must not contain NaN or infinite values")
    return true, pred


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum|y|."""
    actual, forecast = _as_float_arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ValueError("sum of absolute actuals is zero, WAPE is undefined")
    return float(np.abs(actual - forecast).sum() / denominator)
