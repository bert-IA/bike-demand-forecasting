"""Turn the raw hourly bike counts into daily series that respect the data contract."""

from pathlib import Path

import polars as pl

from forecasting.data.ingest_bike import RAW_FILENAME
from forecasting.data.schema import validate_series

TIMEZONE = "Europe/Paris"
PROCESSED_FILENAME = "bike_daily.parquet"


def to_daily_series(raw: pl.LazyFrame) -> pl.DataFrame:
    """Raw hourly counts -> one row per (series_id, day), sorted and validated.

    Pure function: no file access, so it can be tested with tiny DataFrames.
    """
    hourly = raw.select(
        pl.col("id_compteur").alias("series_id"),
        pl.col("date").dt.convert_time_zone(TIMEZONE).alias("timestamp"),
        pl.col("sum_counts").alias("y"),
    )

    daily = (
        hourly.with_columns(pl.col("timestamp").dt.truncate("1d"))
        .group_by("series_id", "timestamp")
        .agg(pl.col("y").sum())
    )

    result = (
        daily.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        .sort("series_id", "timestamp")
        .collect()
    )
    return validate_series(result)


def build_daily_dataset(raw_dir: Path, processed_dir: Path) -> Path:
    """Read the raw Parquet, transform it, write the processed Parquet, return its path."""
    raw = pl.scan_parquet(raw_dir / RAW_FILENAME)
    daily = to_daily_series(raw)

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / PROCESSED_FILENAME
    daily.write_parquet(output_path)
    return output_path


def main() -> None:
    build_daily_dataset(Path("data/raw"), Path("data/processed"))


if __name__ == "__main__":
    main()
