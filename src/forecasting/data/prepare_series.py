import polars as pl

from forecasting.config import (
    MAX_DAYS_MISSING_AT_END,
    MAX_FILLABLE_GAP_DAYS,
    MIN_USEFUL_DAYS,
)


def compute_quality(data: pl.DataFrame) -> pl.DataFrame:
    """Measure data quality per series. Decides nothing."""
    dataset_start = data.select(pl.col("timestamp").min()).item()
    dataset_end = data.select(pl.col("timestamp").max()).item()
    full_span_days = (dataset_end - dataset_start).days + 1

    aggregate = (
        data.group_by("series_id")
        .agg(
            n_observations=pl.len(),
            first_timestamp=pl.col("timestamp").min(),
            last_timestamp=pl.col("timestamp").max(),
            zero_days=(pl.col("y") == 0).sum(),
            first_positive_timestamp=pl.col("timestamp").filter(pl.col("y") > 0).min(),
        )
        .with_columns(
            calendar_span_days=(
                pl.col("last_timestamp") - pl.col("first_timestamp")
            ).dt.total_days()
            + 1
        )
        .with_columns(missing_days_inside=pl.col("calendar_span_days") - pl.col("n_observations"))
        .with_columns(inside_coverage=pl.col("n_observations") / pl.col("calendar_span_days"))
        .with_columns(
            days_missing_at_start=(pl.col("first_timestamp") - dataset_start).dt.total_days()
        )
        .with_columns(days_missing_at_end=(dataset_end - pl.col("last_timestamp")).dt.total_days())
        .with_columns(dataset_coverage=pl.col("n_observations") / full_span_days)
        .with_columns(
            days_before_first_positive=(
                pl.col("first_positive_timestamp") - pl.col("first_timestamp")
            ).dt.total_days()
        )
    )

    with_gaps = data.sort("series_id", "timestamp").with_columns(
        gap=pl.col("timestamp").diff().over("series_id").dt.total_days() - 1
    )

    gaps = with_gaps.group_by("series_id").agg(longest_gap=pl.col("gap").max())

    active = (
        with_gaps.join(aggregate.select("series_id", "first_positive_timestamp"), on="series_id")
        .filter(pl.col("timestamp") >= pl.col("first_positive_timestamp"))
        .sort("series_id", "timestamp")
    )

    runs = (
        active.with_columns(is_zero=pl.col("y") == 0)
        .with_columns(
            new_run=(
                (pl.col("is_zero") != pl.col("is_zero").shift().over("series_id"))
                | (pl.col("gap") > 0)
            ).fill_null(True)
        )
        .with_columns(run_id=pl.col("new_run").cum_sum().over("series_id"))
    )

    longest_zero_run = (
        runs.filter(pl.col("is_zero"))
        .group_by("series_id", "run_id")
        .agg(run_length=pl.len())
        .group_by("series_id")
        .agg(longest_zero_run=pl.col("run_length").max())
    )

    return (
        aggregate.join(gaps, on="series_id", how="left")
        .join(longest_zero_run, on="series_id", how="left")
        .with_columns(pl.col("longest_zero_run").fill_null(0))
        .sort("series_id")
    )


def select_series(
    quality: pl.DataFrame,
    *,
    max_gap_days: int = MAX_FILLABLE_GAP_DAYS,
    max_days_missing_at_end: int = MAX_DAYS_MISSING_AT_END,
    min_useful_days: int = MIN_USEFUL_DAYS,
) -> pl.DataFrame:
    """Return (kept series ids, report of discarded series with their reasons)."""
    return (
        quality.select(
            "series_id",
            dead=pl.col("first_positive_timestamp").is_null(),
            early_stop=pl.col("days_missing_at_end") > max_days_missing_at_end,
            large_gap=pl.col("longest_gap") > max_gap_days,
            too_short=(
                (
                    pl.col("calendar_span_days") - pl.col("days_before_first_positive")
                    < min_useful_days
                ).fill_null(False)
            ),
        )
        .with_columns(kept=~pl.any_horizontal("dead", "early_stop", "large_gap", "too_short"))
        .sort("series_id")
    )
