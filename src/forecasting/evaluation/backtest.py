from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from forecasting.config import INITIAL_TRAIN_DAYS, MAX_FILLABLE_GAP_DAYS, SEASON_LENGTH_DAYS
from forecasting.data.prepare_series import build_daily_grid, compute_quality
from forecasting.evaluation.metrics import (
    mase_from_reference,
    rmsse_from_reference,
    seasonal_naive_reference,
    wape,
)
from forecasting.models.protocol import ForecastModel


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

    if history.select("series_id", "timestamp").is_duplicated().any():
        raise ValueError(
            "duplicate observations in training history: each (series_id, timestamp) must be unique"
        )

    report = build_eligibility_report(history, cutoff)
    selection = report.select(
        "series_id",
        pl.col("eligible").alias("kept"),
    )
    train = build_daily_grid(history, selection)
    return report, train


def run_cutoff(
    data: pl.DataFrame,
    cutoff: datetime,
    model_factory: Callable[[], ForecastModel],
    *,
    horizon: int,
    season_length: int = SEASON_LENGTH_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare data and generate forecasts for one cutoff."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if season_length < 1:
        raise ValueError(f"season_length must be at least 1, got {season_length}")
    report, train = prepare_training_at_cutoff(data, cutoff)
    forecast_tables: list[pl.DataFrame] = []
    reference_rows: list[dict[str, object]] = []
    for current_id in train["series_id"].unique():
        series_train = train.filter(pl.col("series_id") == current_id).sort("timestamp")
        y_train = series_train["y"].to_numpy()
        naive_mae, naive_mse = seasonal_naive_reference(y_train, season_length)
        reference_status = "zero_reference" if naive_mae == 0 and naive_mse == 0 else "ok"
        reference_rows.append(
            {
                "series_id": current_id,
                "cutoff": cutoff,
                "season_length": season_length,
                "naive_mae": naive_mae,
                "naive_mse": naive_mse,
                "reference_status": reference_status,
            }
        )
        model = model_factory()
        model.fit(y_train)
        predictions = model.predict(horizon)

        if predictions.ndim != 1:
            raise ValueError(f"predictions must be 1-D, got {predictions.ndim}")
        if len(predictions) != horizon:
            raise ValueError(f"expected {horizon} predictions, got {len(predictions)}")
        if not np.isfinite(predictions).all():
            raise ValueError("predictions contain non finite values")

        forecast_dates = [cutoff + timedelta(days=day) for day in range(1, horizon + 1)]
        series_predictions = pl.DataFrame(
            {
                "series_id": [current_id] * horizon,
                "cutoff": [cutoff] * horizon,
                "timestamp": forecast_dates,
                "y_pred": predictions,
            }
        )
        forecast_tables.append(series_predictions)

    if forecast_tables:
        forecasts = pl.concat(forecast_tables)
    else:
        forecasts = pl.DataFrame(
            schema={
                "series_id": pl.String,
                "cutoff": pl.Datetime("us"),
                "timestamp": pl.Datetime("us"),
                "y_pred": pl.Float64,
            }
        )

    references = pl.DataFrame(
        reference_rows,
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "season_length": pl.Int64,
            "naive_mae": pl.Float64,
            "naive_mse": pl.Float64,
            "reference_status": pl.String,
        },
    )

    return report, forecasts, references


def run_backtest(
    data: pl.DataFrame,
    cutoffs: list[datetime],
    model_factory: Callable[[], ForecastModel],
    horizon: int,
    season_length: int = SEASON_LENGTH_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run forecasts at each cutoff and return reports, predictions, and references.

    Raise ValueError if no cutoffs are provided.
    """
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("cutoffs must be unique")
    reports: list[pl.DataFrame] = []
    forecasts: list[pl.DataFrame] = []
    references: list[pl.DataFrame] = []
    for cutoff in cutoffs:
        report, forecast, reference = run_cutoff(
            data, cutoff, model_factory, horizon=horizon, season_length=season_length
        )
        report = report.with_columns(cutoff=pl.lit(cutoff))
        reports.append(report)
        forecasts.append(forecast)
        references.append(reference)
    all_reports = pl.concat(reports)
    all_forecasts = pl.concat(forecasts)
    all_references = pl.concat(references)
    return all_reports, all_forecasts, all_references


def attach_actuals_to_forecasts(
    forecasts: pl.DataFrame,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    """Attach observed values to forecasts and flag evaluable rows."""
    if actuals.select("series_id", "timestamp").is_duplicated().any():
        raise ValueError(
            "duplicate observations in actual data: each (series_id, timestamp) must be unique"
        )
    observed = actuals.select("series_id", "timestamp", pl.col("y").alias("y_true"))
    return forecasts.join(observed, on=("series_id", "timestamp"), how="left").with_columns(
        is_evaluable=pl.col("y_true").is_not_null()
    )


def evaluate_wape_by_series(evaluation: pl.DataFrame) -> pl.DataFrame:
    """Calculate WAPE for each series and cutoff."""

    groups = evaluation.partition_by(["series_id", "cutoff"])
    results = []
    for group in groups:
        evaluable = group.filter(pl.col("is_evaluable"))

        n_forecasts = group.height
        n_evaluable = evaluable.height
        if n_evaluable == 0:
            score = None
            status = "no_evaluable_days"
        else:
            for column in ("y_true", "y_pred"):
                values = evaluable[column]
                if values.null_count() > 0 or not values.is_finite().all():
                    raise ValueError(f"{column} must not contain null or non-finite values")

            y_true = evaluable["y_true"].to_numpy()
            y_pred = evaluable["y_pred"].to_numpy()
            if np.abs(y_true).sum() != 0:
                score = wape(y_true, y_pred)
                status = "ok"
            else:
                score = None
                status = "zero_actual_sum"
        results.append(
            {
                "series_id": group["series_id"].item(0),
                "cutoff": group["cutoff"].item(0),
                "n_forecasts": n_forecasts,
                "n_evaluable": n_evaluable,
                "wape": score,
                "wape_status": status,
            }
        )

    return pl.DataFrame(
        results,
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "n_forecasts": pl.Int64,
            "n_evaluable": pl.Int64,
            "wape": pl.Float64,
            "wape_status": pl.String,
        },
    )


def _join_scaled_references(
    evaluation: pl.DataFrame,
    references: pl.DataFrame,
) -> pl.DataFrame:
    keys = ["series_id", "cutoff"]
    if references.select(keys).is_duplicated().any():
        raise ValueError("each series and cutoff must have a unique reference")
    joined = evaluation.join(
        references.with_columns(_reference_found=pl.lit(True)),
        on=keys,
        how="left",
    )
    if joined.filter(pl.col("_reference_found").is_null()).height > 0:
        raise ValueError("a reference is missing for at least one series and cutoff")
    return joined


def _validate_scaled_reference(group: pl.DataFrame) -> tuple[int, float, float]:
    naive_mae = group["naive_mae"].item(0)
    naive_mse = group["naive_mse"].item(0)
    season_length = group["season_length"].item(0)
    reference_status = group["reference_status"].item(0)

    if isinstance(season_length, bool) or not isinstance(season_length, int) or season_length < 1:
        raise ValueError("season_length must be a positive integer")
    if naive_mae is None or not np.isfinite(naive_mae) or naive_mae < 0:
        raise ValueError("naive MAE must be finite and non-negative")
    if naive_mse is None or not np.isfinite(naive_mse) or naive_mse < 0:
        raise ValueError("naive MSE must be finite and non-negative")

    expected_status = "zero_reference" if naive_mae == 0 and naive_mse == 0 else "ok"
    if reference_status != expected_status:
        raise ValueError(f"reference_status must be {expected_status!r}, got {reference_status!r}")
    return season_length, naive_mae, naive_mse


def _evaluate_scaled_group(group: pl.DataFrame) -> dict[str, object]:
    evaluable = group.filter(pl.col("is_evaluable"))
    season_length, naive_mae, naive_mse = _validate_scaled_reference(group)

    if evaluable.is_empty():
        mase_score = None
        rmsse_score = None
        mase_status = "no_evaluable_days"
        rmsse_status = "no_evaluable_days"
    else:
        for column in ("y_true", "y_pred"):
            values = evaluable[column]
            if values.null_count() > 0 or not values.is_finite().all():
                raise ValueError(f"{column} must not contain null or non-finite values")

        y_true = evaluable["y_true"].to_numpy()
        y_pred = evaluable["y_pred"].to_numpy()
        if naive_mae == 0:
            mase_score = None
            mase_status = "zero_reference"
        else:
            mase_score = mase_from_reference(y_true, y_pred, naive_mae)
            mase_status = "ok"

        if naive_mse == 0:
            rmsse_score = None
            rmsse_status = "zero_reference"
        else:
            rmsse_score = rmsse_from_reference(y_true, y_pred, naive_mse)
            rmsse_status = "ok"

    return {
        "series_id": group["series_id"].item(0),
        "cutoff": group["cutoff"].item(0),
        "season_length": season_length,
        "n_forecasts": group.height,
        "n_evaluable": evaluable.height,
        "mase": mase_score,
        "mase_status": mase_status,
        "rmsse": rmsse_score,
        "rmsse_status": rmsse_status,
    }


def evaluate_scaled_metrics_by_series(
    evaluation: pl.DataFrame,
    references: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate MASE and RMSSE for each series and cutoff."""
    joined = _join_scaled_references(evaluation, references)
    results = [
        _evaluate_scaled_group(group) for group in joined.partition_by(["series_id", "cutoff"])
    ]
    return pl.DataFrame(
        results,
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "season_length": pl.Int64,
            "n_forecasts": pl.Int64,
            "n_evaluable": pl.Int64,
            "mase": pl.Float64,
            "mase_status": pl.String,
            "rmsse": pl.Float64,
            "rmsse_status": pl.String,
        },
    )
