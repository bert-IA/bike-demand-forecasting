from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from forecasting.data.prepare_series import compute_quality
from forecasting.evaluation.backtest import (
    Fold,
    active_observations,
    build_eligibility_report,
    prepare_training_at_cutoff,
    rolling_origin_folds,
    summarize_at_cutoff,
)


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


def test_active_observations_excludes_gaps_before_first_positive() -> None:
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

    active = active_observations(data)
    quality_active = compute_quality(active)

    gaps = dict(
        zip(
            quality_active["series_id"],
            quality_active["longest_gap"],
            strict=True,
        )
    )
    assert gaps == {"A": 0, "B": 9}


def test_active_observations_excludes_series_without_positive_value() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "y": [0.0, 0.0],
        }
    )

    active = active_observations(data)

    assert active.height == 0


def test_quality_with_one_active_observation() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [5.0],
        }
    )

    active = active_observations(data)
    quality_active = compute_quality(active)

    assert active.height == 1
    assert quality_active["longest_gap"].item() is None


def test_summarize_at_cutoff_uses_only_available_observations() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "A", "B", "B", "C"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 3),
                datetime(2026, 1, 4),
                datetime(2026, 1, 2),
                datetime(2026, 1, 4),
                datetime(2026, 1, 4),
            ],
            "y": [0.0, 5.0, 8.0, 0.0, 7.0, 3.0],
        }
    )
    cutoff = datetime(2026, 1, 3)

    result = summarize_at_cutoff(data, cutoff).sort("series_id")

    assert result.to_dicts() == [
        {
            "series_id": "A",
            "last_observed_timestamp": datetime(2026, 1, 3),
            "first_positive_timestamp": datetime(2026, 1, 3),
            "days_since_last_observation": 0,
            "useful_days": 1,
        },
        {
            "series_id": "B",
            "last_observed_timestamp": datetime(2026, 1, 2),
            "first_positive_timestamp": None,
            "days_since_last_observation": 1,
            "useful_days": None,
        },
    ]


def test_summarize_at_cutoff_counts_useful_days_inclusively() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "A"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 3),
                datetime(2026, 1, 5),
            ],
            "y": [0.0, 5.0, 7.0],
        }
    )
    cutoff = datetime(2026, 1, 5)

    result = summarize_at_cutoff(data, cutoff)

    assert result["first_positive_timestamp"].item() == datetime(2026, 1, 3)
    assert result["useful_days"].item() == 3


def test_eligibility_report_keeps_series_without_positive_value() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "B"],
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
            "y": [5.0, 0.0],
        }
    )

    report = build_eligibility_report(data, datetime(2026, 1, 1))
    rows = {row["series_id"]: row for row in report.to_dicts()}

    assert rows["A"]["longest_active_gap"] == 0
    assert rows["A"]["too_short"] is True
    assert rows["A"]["eligible"] is False

    assert rows["B"]["longest_active_gap"] is None
    assert rows["B"]["not_started"] is True
    assert rows["B"]["too_short"] is None
    assert rows["B"]["large_gap"] is None
    assert rows["B"]["eligible"] is False


def test_eligibility_report_when_no_series_has_started() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "y": [0.0, 0.0],
        }
    )

    report = build_eligibility_report(data, datetime(2026, 1, 2))

    assert report.height == 1
    assert report["longest_active_gap"].item() is None
    assert report["not_started"].item() is True
    assert report["eligible"].item() is False


@pytest.mark.parametrize(
    ("gap_days", "expected_large_gap", "expected_eligible"),
    [
        pytest.param(7, False, True, id="seven-day-gap-allowed"),
        pytest.param(8, True, False, id="eight-day-gap-rejected"),
    ],
)
def test_eligibility_report_gap_threshold(
    gap_days: int,
    expected_large_gap: bool,
    expected_eligible: bool,
) -> None:
    start = datetime(2026, 1, 1)
    timestamps = [
        start + timedelta(days=day) for day in range(180) if not 50 <= day < 50 + gap_days
    ]
    data = pl.DataFrame(
        {
            "series_id": ["A"] * len(timestamps),
            "timestamp": timestamps,
            "y": [5.0] * len(timestamps),
        }
    )
    cutoff = start + timedelta(days=179)

    report = build_eligibility_report(data, cutoff)

    assert report["useful_days"].item() == 180
    assert report["longest_active_gap"].item() == gap_days
    assert report["large_gap"].item() is expected_large_gap
    assert report["eligible"].item() is expected_eligible


def test_eligibility_report_rejects_missing_cutoff_observation() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    timestamps = [start + timedelta(days=day) for day in range(179)]
    timestamps.append(cutoff + timedelta(days=1))

    data = pl.DataFrame(
        {
            "series_id": ["A"] * len(timestamps),
            "timestamp": timestamps,
            "y": [5.0] * len(timestamps),
        }
    )

    report = build_eligibility_report(data, cutoff)

    assert report["useful_days"].item() == 180
    assert report["days_since_last_observation"].item() == 1
    assert report["missing_at_cutoff"].item() is True
    assert report["eligible"].item() is False


def test_eligibility_report_keeps_multiple_rejection_reasons() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    timestamps = [start + timedelta(days=day) for day in range(179) if not 50 <= day < 58]
    data = pl.DataFrame(
        {
            "series_id": ["A"] * len(timestamps),
            "timestamp": timestamps,
            "y": [5.0] * len(timestamps),
        }
    )

    report = build_eligibility_report(data, cutoff)

    assert report["useful_days"].item() == 180
    assert report["too_short"].item() is False
    assert report["longest_active_gap"].item() == 8
    assert report["large_gap"].item() is True
    assert report["missing_at_cutoff"].item() is True
    assert report["eligible"].item() is False


def test_eligibility_report_uses_active_history_only() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 11),
                datetime(2026, 1, 12),
                datetime(2026, 1, 1),
                datetime(2026, 1, 11),
                datetime(2026, 1, 12),
                datetime(2026, 1, 1),
                datetime(2026, 1, 13),
            ],
            "y": [0.0, 5.0, 6.0, 4.0, 5.0, 6.0, 0.0, 3.0],
        }
    )
    cutoff = datetime(2026, 1, 12)

    report = build_eligibility_report(data, cutoff)
    rows = {row["series_id"]: row for row in report.to_dicts()}

    assert rows["A"]["longest_active_gap"] == 0
    assert rows["A"]["large_gap"] is False

    assert rows["B"]["longest_active_gap"] == 9
    assert rows["B"]["large_gap"] is True

    assert rows["C"]["not_started"] is True
    assert rows["C"]["longest_active_gap"] is None
    assert rows["C"]["eligible"] is False


def test_eligibility_report_before_first_observation() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 2)],
            "y": [5.0],
        }
    )

    report = build_eligibility_report(data, datetime(2026, 1, 1))

    assert report.height == 0
    assert "series_id" in report.columns
    assert "longest_active_gap" in report.columns
    assert "eligible" in report.columns


def test_prepare_training_at_cutoff_ignores_future_values() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    days = [day for day in range(181) if day != 90]
    data = pl.DataFrame(
        {
            "series_id": ["A"] * len(days),
            "timestamp": [start + timedelta(days=day) for day in days],
            "y": [float(day + 1) for day in days],
        }
    )

    report, train = prepare_training_at_cutoff(data, cutoff)

    missing_day = train.filter(pl.col("timestamp") == start + timedelta(days=90))

    assert report["eligible"].item() is True
    assert train.height == 180
    assert train["timestamp"].max() == cutoff
    assert missing_day["y"].item() == 91.0
    assert missing_day["is_imputed"].item() is True

    modified_data = data.with_columns(
        pl.when(pl.col("timestamp") > cutoff).then(999999.0).otherwise(pl.col("y")).alias("y")
    )

    modified_report, modified_train = prepare_training_at_cutoff(modified_data, cutoff)

    assert_frame_equal(report, modified_report)
    assert_frame_equal(train, modified_train)


def test_prepare_training_at_cutoff_when_no_series_is_eligible() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "y": [5.0, 6.0],
        }
    )
    cutoff = datetime(2026, 1, 2)

    report, train = prepare_training_at_cutoff(data, cutoff)

    assert report.height == 1
    assert report["series_id"].item() == "A"
    assert report["too_short"].item() is True
    assert report["eligible"].item() is False

    assert train.height == 0
    assert train.columns == [
        "series_id",
        "timestamp",
        "y",
        "is_imputed",
        "is_long_zero_run",
    ]


def test_prepare_training_at_cutoff_keeps_only_eligible_series() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    rows = [
        {
            "series_id": "A",
            "timestamp": start + timedelta(days=day),
            "y": 5.0,
        }
        for day in range(180)
    ]
    rows.extend(
        [
            {"series_id": "B", "timestamp": start, "y": 3.0},
            {"series_id": "B", "timestamp": cutoff, "y": 4.0},
        ]
    )
    data = pl.DataFrame(rows)

    report, train = prepare_training_at_cutoff(data, cutoff)
    eligibility = dict(zip(report["series_id"], report["eligible"], strict=True))

    assert eligibility == {"A": True, "B": False}
    assert set(train["series_id"]) == {"A"}
    assert train.height == 180
