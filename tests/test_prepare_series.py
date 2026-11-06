from datetime import datetime
from typing import Any

import polars as pl
import pytest

from forecasting.data.prepare_series import build_daily_grid, compute_quality, select_series

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


def row(table: pl.DataFrame, series_id: str) -> dict[str, Any]:
    return table.filter(pl.col("series_id") == series_id).row(0, named=True)


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


# --- select_series ---
@pytest.fixture
def selection(quality: pl.DataFrame) -> pl.DataFrame:
    return select_series(quality, max_gap_days=2, max_days_missing_at_end=3, min_useful_days=6)


@pytest.mark.parametrize(
    ("series_id", "dead", "early_stop", "large_gap", "too_short", "kept"),
    [
        pytest.param("A", False, False, False, False, True, id="a-kept"),
        pytest.param("B", False, False, True, False, False, id="b-large-gap"),
        pytest.param("C", False, False, False, False, True, id="c-kept"),
        pytest.param("D", True, True, False, False, False, id="d-dead"),
        pytest.param("E", False, False, False, True, False, id="e-too-short"),
        pytest.param("F", False, True, False, False, False, id="f-early-stop"),
        pytest.param("G", False, True, False, True, False, id="g-two-reasons"),
    ],
)
def test_selection_flags(
    selection: pl.DataFrame,
    series_id: str,
    dead: bool,
    early_stop: bool,
    large_gap: bool,
    too_short: bool,
    kept: bool,
) -> None:
    assert row(selection, series_id) == {
        "series_id": series_id,
        "dead": dead,
        "early_stop": early_stop,
        "large_gap": large_gap,
        "too_short": too_short,
        "kept": kept,
    }


def test_too_short_threshold_is_strict(quality: pl.DataFrame) -> None:
    kept_at_limit = select_series(
        quality, max_gap_days=2, max_days_missing_at_end=3, min_useful_days=5
    )
    rejected_below = select_series(
        quality, max_gap_days=2, max_days_missing_at_end=3, min_useful_days=6
    )
    assert row(kept_at_limit, "E")["too_short"] is False
    assert row(rejected_below, "E")["too_short"] is True


# {series_id: {day_of_january: y}}. A missing day means no row at all.
GRID_SERIES: dict[str, dict[int, int]] = {
    "F": {1: 4, 2: 0, 3: 0, 6: 0, 7: 5},
    "G": dict(zip(range(1, 9), [0, 0, 0, 5, 6, 0, 0, 7], strict=True)),
    "H": dict(zip(range(1, 11), [3, *[0] * 7, 4, 5], strict=True)),
    "I": dict(zip(range(1, 12), [3, *[0] * 8, 4, 5], strict=True)),
    "J": {1: 0, 2: 5, 3: 0, 5: 7, 6: 4},
    "Z": {1: 5, 2: 6, 3: 7},
}
KEPT = {"F": True, "G": True, "H": True, "I": True, "J": True, "Z": False}


def make_grid_input() -> pl.DataFrame:
    rows = [
        {"series_id": sid, "timestamp": datetime(2026, 1, day), "y": float(y)}
        for sid, values in GRID_SERIES.items()
        for day, y in values.items()
    ]
    return pl.DataFrame(rows).sort("timestamp", descending=True)


def make_grid_selection() -> pl.DataFrame:
    return pl.DataFrame({"series_id": list(KEPT), "kept": list(KEPT.values())})


@pytest.fixture
def grid() -> pl.DataFrame:
    return build_daily_grid(make_grid_input(), make_grid_selection())


# (day_of_january, y, is_imputed, is_long_zero_run)
GridRow = tuple[int, float, bool, bool]
EXPECTED_GRID: dict[str, list[GridRow]] = {
    "F": [
        (1, 4.0, False, False),
        (2, 0.0, False, False),
        (3, 0.0, False, False),
        (4, 0.0, True, False),
        (5, 0.0, True, False),
        (6, 0.0, False, False),
        (7, 5.0, False, False),
    ],
    "G": [
        (4, 5.0, False, False),
        (5, 6.0, False, False),
        (6, 0.0, False, False),
        (7, 0.0, False, False),
        (8, 7.0, False, False),
    ],
    "H": [
        (1, 3.0, False, False),
        *[(day, 0.0, False, False) for day in range(2, 9)],
        (9, 4.0, False, False),
        (10, 5.0, False, False),
    ],
    "I": [
        (1, 3.0, False, False),
        *[(day, 0.0, False, True) for day in range(2, 10)],
        (10, 4.0, False, False),
        (11, 5.0, False, False),
    ],
    "J": [
        (2, 5.0, False, False),
        (3, 0.0, False, False),
        (4, 3.5, True, False),
        (5, 7.0, False, False),
        (6, 4.0, False, False),
    ],
}


@pytest.mark.parametrize("series_id", list(EXPECTED_GRID))
def test_grid_rows(grid: pl.DataFrame, series_id: str) -> None:
    actual = (
        grid.filter(pl.col("series_id") == series_id)
        .sort("timestamp")
        .select(pl.col("timestamp").dt.day(), "y", "is_imputed", "is_long_zero_run")
        .rows()
    )
    assert actual == EXPECTED_GRID[series_id]


def test_grid_drops_unkept_series(grid: pl.DataFrame) -> None:
    assert set(grid["series_id"]) == {"F", "G", "H", "I", "J"}


def test_grid_has_no_null(grid: pl.DataFrame) -> None:
    assert grid["y"].null_count() == 0


def test_grid_columns(grid: pl.DataFrame) -> None:
    assert grid.columns == ["series_id", "timestamp", "y", "is_imputed", "is_long_zero_run"]


# mise à jour des decisions de selections des compteurs
def test_quality_uses_last_observed_day_as_dataset_end() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "B", "B"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
            ],
            "y": [10.0, 12.0, 20.0, 22.0],
        }
    )
    cutoff = datetime(2026, 1, 3)

    quality = compute_quality(data)

    for series_id in ("A", "B"):
        series = row(quality, series_id)

        assert series["last_timestamp"] == datetime(2026, 1, 2)
        assert series["days_missing_at_end"] == 0
        assert (cutoff - series["last_timestamp"]).days == 1


def test_quality_longest_gap_includes_days_before_first_positive() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "A", "B", "B", "B"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 11),
                datetime(2026, 1, 12),
                datetime(2026, 1, 1),
                datetime(2026, 1, 11),
                datetime(2026, 1, 12),
            ],
            "y": [0.0, 5.0, 6.0, 4.0, 5.0, 6.0],
        }
    )

    quality = compute_quality(data)

    assert row(quality, "A")["first_positive_timestamp"] == datetime(2026, 1, 11)
    assert row(quality, "A")["longest_gap"] == 9

    assert row(quality, "B")["first_positive_timestamp"] == datetime(2026, 1, 1)
    assert row(quality, "B")["longest_gap"] == 9
