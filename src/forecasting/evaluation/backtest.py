from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_end: int  # indice exclu : le train est [0, train_end)
    test_end: int  # le test est [train_end, test_end)


def rolling_origin_folds(n_obs: int, initial_train: int, horizon: int, step: int) -> list[Fold]:
    """Return expanding-window folds; train is [0, train_end), test is [train_end, test_end).
    An incomplete last test window is dropped."""
    for name, value in (("initial_train", initial_train), ("horizon", horizon), ("step", step)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if initial_train + horizon > n_obs:
        raise ValueError(
            f"n_obs={n_obs} is too short for initial_train={initial_train} and horizon={horizon} : "
            f"{initial_train + horizon} > {n_obs}"
        )
    folds: list[Fold] = []
    train_end = initial_train
    while train_end + horizon <= n_obs:
        folds.append(Fold(train_end, train_end + horizon))
        train_end += step
    return folds
