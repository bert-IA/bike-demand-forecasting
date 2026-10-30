import pytest

from forecasting.evaluation.backtest import Fold, rolling_origin_folds


def test_rolling_origin_folds_hand_computed() -> None:
    # Train grows by 2 days per fold; each test covers the 2 days after the train
    assert rolling_origin_folds(n_obs=10, initial_train=4, horizon=2, step=2) == [
        Fold(4, 6),
        Fold(6, 8),
        Fold(8, 10),
    ]


def test_rolling_origin_folds_incomplete_last_fold() -> None:
    # The third test window would need position 9 (n_obs=9), so it is dropped.
    assert rolling_origin_folds(n_obs=9, initial_train=4, horizon=2, step=2) == [
        Fold(4, 6),
        Fold(6, 8),
    ]


def test_rolling_origin_folds_series_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        rolling_origin_folds(n_obs=5, initial_train=4, horizon=2, step=2)


@pytest.mark.parametrize(
    ("initial_train", "horizon", "step"),
    [
        pytest.param(0, 2, 2, id="initial-train-zero"),
        pytest.param(4, 0, 2, id="horizon-zero"),
        pytest.param(4, 2, 0, id="step-zero"),
    ],
)
def test_rolling_origin_folds_invalid_parameters_raise(
    initial_train: int, horizon: int, step: int
) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        rolling_origin_folds(n_obs=10, initial_train=initial_train, horizon=horizon, step=step)


def test_rolling_origin_folds_step_and_horizon_distinct() -> None:
    assert rolling_origin_folds(n_obs=6, initial_train=3, horizon=2, step=1) == [
        Fold(3, 5),
        Fold(4, 6),
    ]
