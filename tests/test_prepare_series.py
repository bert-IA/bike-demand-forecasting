from datetime import datetime
from typing import Any

import polars as pl
import pytest

from forecasting.data.prepare_series import compute_quality

# {series_id: {day_of_january: y}}. A missing day means no row at all.
SAMPLE_SERIES: dict[str, dict[int, int]] = {
    "A": dict(zip(range(1, 11), [5, 6, 7, 8, 9, 4, 3, 5, 6, 7], strict=True)),
    "B": {1: 12, 2: 14, 3: 13, 7: 15, 8: 0, 10: 14, 11: 16, 12: 13},
    "C": dict(zip(range(1, 13), [4, 5, 0, 0, 0, 6, 0, 0, 3, 4, 0, 5], strict=True)),
    "D": dict(zip(range(1, 7), [0] * 6, strict=True)),
    "E": dict(zip(range(8, 13), [3, 4, 5, 2, 3], strict=True)),
    "F": {1: 4, 2: 0, 3: 0, 6: 0, 7: 5},
    "G": dict(zip(range(1, 9), [0, 0, 0, 5, 6, 0, 0, 7], strict=True)),
}
DATASET_DAYS = 12


def make_sample() -> pl.DataFrame:
    """Flatten SAMPLE_SERIES into one row per (series_id, day)."""
    rows = [
        (series_id, datetime(2025, 1, day), y)
        for series_id, days in SAMPLE_SERIES.items()
        for day, y in days.items()
    ]
    return pl.DataFrame(
        rows,
        schema={"series_id": pl.String, "timestamp": pl.Datetime, "y": pl.Int64},
        orient="row",
    )


@pytest.fixture
def quality() -> pl.DataFrame:
    return compute_quality(make_sample())


def row(quality: pl.DataFrame, series_id: str) -> dict[str, Any]:
    return quality.filter(pl.col("series_id") == series_id).row(0, named=True)


def test_keeps_one_row_per_series(quality: pl.DataFrame) -> None:
    assert quality.height == len(SAMPLE_SERIES)


@pytest.mark.parametrize("series_id", list(SAMPLE_SERIES))
def test_absent_days_identity(quality: pl.DataFrame, series_id: str) -> None:
    r = row(quality, series_id)
    total = (
        r["days_missing_at_start"]
        + r["missing_days_inside"]
        + r["days_missing_at_end"]
        + r["n_observations"]
    )
    assert total == DATASET_DAYS


@pytest.mark.parametrize(
    ("series_id", "column", "expected"),
    [
        pytest.param("B", "longest_gap", 3, id="B-longest-gap"),
        pytest.param("B", "missing_days_inside", 4, id="B-missing-inside"),
        pytest.param("F", "longest_gap", 2, id="F-longest-gap"),
        pytest.param("C", "longest_zero_run", 3, id="C-zero-run"),
        pytest.param("F", "longest_zero_run", 2, id="F-gap-splits-zero-runs"),
        pytest.param("G", "longest_zero_run", 2, id="G-leading-zeros-ignored"),
        pytest.param("A", "longest_zero_run", 0, id="A-no-zero-run"),
        pytest.param("D", "longest_zero_run", 0, id="D-dead-counter"),
        pytest.param("E", "days_missing_at_start", 7, id="E-late-start"),
        pytest.param("F", "days_missing_at_end", 5, id="F-early-stop"),
        pytest.param("G", "days_before_first_positive", 3, id="G-leading-zeros"),
        pytest.param("D", "first_positive_timestamp", None, id="D-never-positive"),
    ],
)
def test_expected_values(
    quality: pl.DataFrame, series_id: str, column: str, expected: object
) -> None:
    assert row(quality, series_id)[column] == expected
