"""Tests for the time series data contract."""

from datetime import datetime

import pandera.errors
import polars as pl
import pytest

from forecasting.data.schema import validate_series


def make_df(
    series_ids: list[str | None],
    timestamps: list[datetime],
    values: list[float],
) -> pl.DataFrame:
    """Build a small DataFrame so each test only states what makes it special."""
    return pl.DataFrame({"series_id": series_ids, "timestamp": timestamps, "y": values})


def test_valid_dataframe_passes() -> None:
    # Arrange: two series (A and B), each sorted, no duplicate, y >= 0.
    df = make_df(
        ["A", "A", "B"],
        [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 1)],
        [10.0, 12.0, 3.0],
    )
    # Act: a valid table must go through without any exception.
    result = validate_series(df)
    # Assert: validation checks, it never drops or adds rows.
    assert result.height == df.height


def test_negative_y_is_rejected() -> None:
    df = make_df(["A", "A"], [datetime(2024, 1, 1), datetime(2024, 1, 2)], [10.0, -1.0])
    # The `ge=0` rule on y is checked by Pandera, which raises SchemaError.
    # The test passes only if this exception is raised inside the `with` block.
    with pytest.raises(pandera.errors.SchemaError):
        validate_series(df)


def test_duplicate_timestamp_in_series_is_rejected() -> None:
    # Series A has the same timestamp twice: forbidden by `unique` in the schema.
    df = make_df(["A", "A"], [datetime(2024, 1, 1), datetime(2024, 1, 1)], [10.0, 11.0])
    with pytest.raises(pandera.errors.SchemaError):
        validate_series(df)


def test_null_series_id_is_rejected() -> None:
    # An observation without series identifier is unusable: nullable=False.
    # This is why make_df accepts `str | None`.
    df = make_df(["A", None], [datetime(2024, 1, 1), datetime(2024, 1, 2)], [10.0, 11.0])
    with pytest.raises(pandera.errors.SchemaError):
        validate_series(df)


def test_unsorted_timestamps_are_rejected() -> None:
    # Series A goes backwards in time (Jan 2 then Jan 1): our own check, not Pandera's,
    # so the exception is a plain ValueError.
    df = make_df(["A", "A"], [datetime(2024, 1, 2), datetime(2024, 1, 1)], [10.0, 11.0])
    # `match` is a regex searched in the error message: it checks we fail for the right
    # reason, not just that something failed.
    with pytest.raises(ValueError, match="increasing"):
        validate_series(df)


def test_decreasing_order_is_per_series() -> None:
    # A ends on Jan 5, then B starts on Jan 1. Compared globally the time goes backwards
    # between rows 2 and 3, but each series is sorted on its own: this must PASS.
    # If `.over("series_id")` were forgotten, this test would fail.
    df = make_df(
        ["A", "A", "B", "B"],
        [datetime(2024, 1, 4), datetime(2024, 1, 5), datetime(2024, 1, 1), datetime(2024, 1, 2)],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert validate_series(df).height == 4
