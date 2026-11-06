from dataclasses import dataclass
from datetime import datetime

import polars as pl

from forecasting.config import INITIAL_TRAIN_DAYS, MAX_FILLABLE_GAP_DAYS
from forecasting.data.prepare_series import build_daily_grid, compute_quality


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


def active_observations(data: pl.DataFrame) -> pl.DataFrame:
    """Keep observations from the first positive value of each series."""
    return (
        data.with_columns(
            first_positive_timestamp=pl.col("timestamp")
            .filter(pl.col("y") > 0)
            .min()
            .over("series_id")
        )
        .filter(pl.col("timestamp") >= pl.col("first_positive_timestamp"))
        .drop("first_positive_timestamp")
    )


def summarize_at_cutoff(data: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Summarize each series using observations up to the cutoff."""
    return (
        data.filter(pl.col("timestamp") <= cutoff)
        .group_by("series_id")
        .agg(
            last_observed_timestamp=pl.col("timestamp").max(),
            first_positive_timestamp=pl.col("timestamp").filter(pl.col("y") > 0).min(),
            days_since_last_observation=(
                pl.lit(cutoff) - pl.col("timestamp").max()
            ).dt.total_days(),
        )
        .with_columns(
            useful_days=(pl.lit(cutoff) - pl.col("first_positive_timestamp")).dt.total_days() + 1
        )
    )


def build_eligibility_report(data: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Build a series eligibility report using observations up to the cutoff."""
    history = data.filter(pl.col("timestamp") <= cutoff)
    summary = summarize_at_cutoff(history, cutoff)
    active = active_observations(history)

    if active.is_empty():
        report = summary.with_columns(longest_active_gap=pl.lit(None, dtype=pl.Int64))
    else:
        quality_active = compute_quality(active)

        gaps = quality_active.select(
            "series_id",
            pl.col("longest_gap").fill_null(0).alias("longest_active_gap"),
        )

        report = summary.join(gaps, on="series_id", how="left")

    return (
        report.with_columns(
            not_started=pl.col("first_positive_timestamp").is_null(),
            too_short=pl.col("useful_days") < INITIAL_TRAIN_DAYS,
            missing_at_cutoff=pl.col("days_since_last_observation") > 0,
            large_gap=pl.col("longest_active_gap") > MAX_FILLABLE_GAP_DAYS,
        )
        .with_columns(
            eligible=~pl.any_horizontal(
                "not_started",
                pl.col("too_short").fill_null(False),
                "missing_at_cutoff",
                pl.col("large_gap").fill_null(False),
            )
        )
        .sort("series_id")
    )


def prepare_training_at_cutoff(
    data: pl.DataFrame,
    cutoff: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the eligibility report and prepared training data at the cutoff."""
    history = data.filter(pl.col("timestamp") <= cutoff)
    report = build_eligibility_report(history, cutoff)
    selection = report.select(
        "series_id",
        pl.col("eligible").alias("kept"),
    )
    train = build_daily_grid(history, selection)
    return report, train
