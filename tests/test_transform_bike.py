"""Tests for the hourly -> daily transformation (no files, no network)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from forecasting.data.ingest_bike import RAW_FILENAME
from forecasting.data.transform_bike import PROCESSED_FILENAME, build_daily_dataset, to_daily_series


def make_raw(rows: list[tuple[str, datetime, float | None]]) -> pl.LazyFrame:
    """Build a raw-like LazyFrame with the columns the transform needs, plus one useless one."""
    return pl.DataFrame(
        {
            "id_compteur": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            "sum_counts": [r[2] for r in rows],
            "url_photos_n1": ["ignored"] * len(rows),
        },
        schema_overrides={"date": pl.Datetime("ms", "UTC")},
    ).lazy()


def test_output_columns_and_names() -> None:
    raw = make_raw([("A", datetime(2026, 3, 10, 12, tzinfo=UTC), 5.0)])

    result = to_daily_series(raw)

    assert result.columns == ["series_id", "timestamp", "y"]


def test_hours_of_same_day_are_summed() -> None:
    raw = make_raw(
        [
            ("A", datetime(2026, 3, 10, 10, tzinfo=UTC), 2.0),
            ("A", datetime(2026, 3, 10, 11, tzinfo=UTC), 3.0),
        ]
    )

    result = to_daily_series(raw)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["series_id"] == "A"
    assert row["y"] == 5.0
    assert row["timestamp"] == datetime(2026, 3, 10)


def test_late_utc_hour_moves_to_next_paris_day() -> None:
    # 23:30 UTC on 03-10 is 00:30 on 03-11 in Paris (UTC+1).
    raw = make_raw([("A", datetime(2026, 3, 10, 23, 30, tzinfo=UTC), 5.0)])

    result = to_daily_series(raw)

    assert result["timestamp"][0] == datetime(2026, 3, 11)


def test_daylight_saving_day_is_a_single_day() -> None:
    # On 2026-10-25 Paris has 25 hours: all 25 hourly rows belong to the same local day.
    start = datetime(2026, 10, 24, 22, tzinfo=UTC)
    raw = make_raw([("A", start + timedelta(hours=i), 1.0) for i in range(25)])

    result = to_daily_series(raw)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["timestamp"] == datetime(2026, 10, 25)
    assert row["y"] == 25.0


def test_series_are_kept_separate_and_sorted() -> None:
    raw = make_raw(
        [
            ("B", datetime(2026, 3, 10, 10, tzinfo=UTC), 2.0),
            ("A", datetime(2026, 3, 11, 11, tzinfo=UTC), 3.0),
        ]
    )

    result = to_daily_series(raw)

    assert result["series_id"].to_list() == ["A", "B"]


def test_negative_counts_are_rejected() -> None:
    raw = make_raw([("A", datetime(2026, 3, 10, 12, tzinfo=UTC), -1.0)])

    with pytest.raises(pandera.errors.SchemaError):
        to_daily_series(raw)


def test_build_daily_dataset_writes_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_raw([("A", datetime(2026, 3, 10, 12, tzinfo=UTC), 5.0)]).collect().write_parquet(
        raw_dir / RAW_FILENAME
    )

    output = build_daily_dataset(raw_dir, tmp_path / "processed")

    assert output == tmp_path / "processed" / PROCESSED_FILENAME
    assert pl.read_parquet(output).height == 1
