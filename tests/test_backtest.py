from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from polars.testing import assert_frame_equal

from forecasting.data.prepare_series import compute_quality
from forecasting.evaluation.backtest import (
    Fold,
    active_observations,
    attach_actuals_to_forecasts,
    build_eligibility_report,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
    prepare_training_at_cutoff,
    rolling_origin_folds,
    run_backtest,
    run_cutoff,
    summarize_at_cutoff,
)
from forecasting.models.protocol import ForecastModel


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


def test_run_cutoff_predicts_each_eligible_series() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    rows = [
        {
            "series_id": series_id,
            "timestamp": start + timedelta(days=day),
            "y": value,
        }
        for series_id, value in [("A", 10.0), ("B", 20.0)]
        for day in range(180)
    ]
    data = pl.DataFrame(rows)

    class FakeModel:
        def __init__(self) -> None:
            self.last_value = 0.0

        def fit(self, train: NDArray[np.float64]) -> None:
            self.last_value = float(train[-1])

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, self.last_value, dtype=np.float64)

    models: list[FakeModel] = []

    def make_model() -> FakeModel:
        model = FakeModel()
        models.append(model)
        return model

    report, predictions, references = run_cutoff(data, cutoff, make_model, horizon=2)

    assert report["eligible"].sum() == 2
    assert len(models) == 2
    assert models[0] is not models[1]
    assert references.sort("series_id").to_dicts() == [
        {
            "series_id": "A",
            "cutoff": cutoff,
            "season_length": 7,
            "naive_mae": 0.0,
            "naive_mse": 0.0,
            "reference_status": "zero_reference",
        },
        {
            "series_id": "B",
            "cutoff": cutoff,
            "season_length": 7,
            "naive_mae": 0.0,
            "naive_mse": 0.0,
            "reference_status": "zero_reference",
        },
    ]

    assert predictions.sort("series_id", "timestamp").to_dicts() == [
        {
            "series_id": "A",
            "cutoff": cutoff,
            "timestamp": cutoff + timedelta(days=1),
            "y_pred": 10.0,
        },
        {
            "series_id": "A",
            "cutoff": cutoff,
            "timestamp": cutoff + timedelta(days=2),
            "y_pred": 10.0,
        },
        {
            "series_id": "B",
            "cutoff": cutoff,
            "timestamp": cutoff + timedelta(days=1),
            "y_pred": 20.0,
        },
        {
            "series_id": "B",
            "cutoff": cutoff,
            "timestamp": cutoff + timedelta(days=2),
            "y_pred": 20.0,
        },
    ]


def test_run_cutoff_without_eligible_series() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [5.0],
        }
    )
    cutoff = datetime(2026, 1, 1)

    def make_model() -> ForecastModel:
        raise AssertionError("A model should not be created")

    report, predictions, references = run_cutoff(data, cutoff, make_model, horizon=2)

    assert report["eligible"].item() is False
    assert predictions.is_empty()
    assert predictions.columns == ["series_id", "cutoff", "timestamp", "y_pred"]
    assert predictions.schema == {
        "series_id": pl.String,
        "cutoff": pl.Datetime("us"),
        "timestamp": pl.Datetime("us"),
        "y_pred": pl.Float64,
    }
    assert references.is_empty()
    assert references.schema == {
        "series_id": pl.String,
        "cutoff": pl.Datetime("us"),
        "season_length": pl.Int64,
        "naive_mae": pl.Float64,
        "naive_mse": pl.Float64,
        "reference_status": pl.String,
    }


def test_run_cutoff_rejects_wrong_number_of_predictions() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [5.0] * 180,
        }
    )

    class WrongLengthModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.array([5.0], dtype=np.float64)

    with pytest.raises(ValueError, match="expected 2 predictions"):
        run_cutoff(data, cutoff, WrongLengthModel, horizon=2)


def test_run_cutoff_rejects_predictions_with_wrong_shape() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [5.0] * 180,
        }
    )

    class WrongShapeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="predictions must be 1-D"):
        run_cutoff(data, cutoff, WrongShapeModel, horizon=2)


def test_run_cutoff_uses_custom_season_length() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)
    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [float(day + 1) for day in range(180)],
        }
    )

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.zeros(horizon, dtype=np.float64)

    _, _, references = run_cutoff(data, cutoff, FakeModel, horizon=2, season_length=3)

    assert references.to_dicts() == [
        {
            "series_id": "A",
            "cutoff": cutoff,
            "season_length": 3,
            "naive_mae": 3.0,
            "naive_mse": 9.0,
            "reference_status": "ok",
        }
    ]


@pytest.mark.parametrize("season_length", [0, -1])
def test_run_cutoff_rejects_invalid_season_without_eligible_series(
    season_length: int,
) -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [5.0],
        }
    )
    cutoff = datetime(2026, 1, 1)

    def make_model() -> ForecastModel:
        raise AssertionError("A model should not be created")

    with pytest.raises(ValueError, match="season_length must be at least 1"):
        run_cutoff(data, cutoff, make_model, horizon=2, season_length=season_length)


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
    ],
)
def test_run_cutoff_rejects_non_finite_predictions(invalid_value: float) -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [5.0] * 180,
        }
    )

    class NonFiniteModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.array([5.0, invalid_value], dtype=np.float64)

    with pytest.raises(ValueError, match="predictions contain non finite values"):
        run_cutoff(data, cutoff, NonFiniteModel, horizon=2)


@pytest.mark.parametrize("horizon", [0, -1])
def test_run_cutoff_rejects_invalid_horizon(horizon: int) -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [5.0],
        }
    )

    def make_model() -> ForecastModel:
        raise AssertionError("A model should not be created")

    with pytest.raises(ValueError, match="horizon must be at least 1"):
        run_cutoff(data, datetime(2026, 1, 1), make_model, horizon=horizon)


def test_run_cutoff_rejects_wrong_dim_0() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [5.0] * 180,
        }
    )

    class ScalarPredictionModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.array(5.0, dtype=np.float64)

    with pytest.raises(ValueError, match="predictions must be 1-D"):
        run_cutoff(data, cutoff, ScalarPredictionModel, horizon=2)


def test_run_backtest_rejects_empty_cutoffs() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [5.0],
        }
    )

    def make_model() -> ForecastModel:
        raise AssertionError("A model should not be created")

    with pytest.raises(ValueError, match="cutoffs must not be empty"):
        run_backtest(data, [], make_model, horizon=2)


def test_run_backtest_combines_cutoffs() -> None:
    start = datetime(2026, 1, 1)
    first_cutoff = start + timedelta(days=178)
    second_cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [5.0] * 180,
        }
    )

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, 5.0, dtype=np.float64)

    reports, forecasts, references = run_backtest(
        data, [first_cutoff, second_cutoff], FakeModel, horizon=2
    )

    assert reports.sort("cutoff").select("cutoff", "eligible").rows() == [
        (first_cutoff, False),
        (second_cutoff, True),
    ]

    assert forecasts.sort("timestamp").to_dicts() == [
        {
            "series_id": "A",
            "cutoff": second_cutoff,
            "timestamp": second_cutoff + timedelta(days=1),
            "y_pred": 5.0,
        },
        {
            "series_id": "A",
            "cutoff": second_cutoff,
            "timestamp": second_cutoff + timedelta(days=2),
            "y_pred": 5.0,
        },
    ]


def test_run_backtest_trains_a_new_model_at_each_cutoff() -> None:
    start = datetime(2026, 1, 1)
    first_cutoff = start + timedelta(days=179)
    second_cutoff = start + timedelta(days=180)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 181,
            "timestamp": [start + timedelta(days=day) for day in range(181)],
            "y": [5.0] * 180 + [9.0],
        }
    )

    class FakeModel:
        def __init__(self) -> None:
            self.last_value = 0.0

        def fit(self, train: NDArray[np.float64]) -> None:
            self.last_value = float(train[-1])

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, self.last_value, dtype=np.float64)

    models: list[FakeModel] = []

    def make_model() -> FakeModel:
        model = FakeModel()
        models.append(model)
        return model

    reports, forecasts, references = run_backtest(
        data, [first_cutoff, second_cutoff], make_model, horizon=1
    )

    assert reports["eligible"].to_list() == [True, True]
    assert len(models) == 2
    assert models[0] is not models[1]
    assert forecasts.sort("cutoff").select("cutoff", "y_pred").rows() == [
        (first_cutoff, 5.0),
        (second_cutoff, 9.0),
    ]


def test_run_backtest_keeps_references_for_each_cutoff() -> None:
    start = datetime(2026, 1, 1)
    first_cutoff = start + timedelta(days=179)
    second_cutoff = start + timedelta(days=180)

    values = [float(day + 1) for day in range(180)] + [190.0]
    data = pl.DataFrame(
        {
            "series_id": ["A"] * len(values),
            "timestamp": [start + timedelta(days=day) for day in range(len(values))],
            "y": values,
        }
    )

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.zeros(horizon, dtype=np.float64)

    _, _, references = run_backtest(
        data,
        [first_cutoff, second_cutoff],
        FakeModel,
        horizon=2,
        season_length=3,
    )

    rows = references.sort("cutoff").to_dicts()

    assert len(rows) == 2
    assert [row["cutoff"] for row in rows] == [first_cutoff, second_cutoff]
    assert [row["season_length"] for row in rows] == [3, 3]
    assert [row["reference_status"] for row in rows] == ["ok", "ok"]
    assert rows[0]["naive_mae"] == pytest.approx(3.0)
    assert rows[0]["naive_mse"] == pytest.approx(9.0)
    assert rows[1]["naive_mae"] == pytest.approx(543 / 178)
    assert rows[1]["naive_mse"] == pytest.approx(1737 / 178)


def test_run_backtest_combines_empty_and_nonempty_references() -> None:
    start = datetime(2026, 1, 1)
    first_cutoff = start + timedelta(days=178)
    second_cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)],
            "y": [float(day + 1) for day in range(180)],
        }
    )

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.zeros(horizon, dtype=np.float64)

    reports, forecasts, references = run_backtest(
        data,
        [first_cutoff, second_cutoff],
        FakeModel,
        horizon=2,
        season_length=3,
    )

    assert reports.sort("cutoff")["eligible"].to_list() == [False, True]
    assert forecasts.height == 2
    assert references.height == 1
    assert references["cutoff"].item() == second_cutoff
    assert references["naive_mae"].item() == pytest.approx(3.0)


def test_attach_actuals_keeps_forecasts_without_observations() -> None:
    cutoff = datetime(2026, 1, 1)
    forecasts = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [cutoff, cutoff],
            "timestamp": [
                datetime(2026, 1, 2),
                datetime(2026, 1, 3),
            ],
            "y_pred": [10.0, 11.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 2)],
            "y": [12.0],
        }
    )

    result = attach_actuals_to_forecasts(forecasts, actuals).sort("timestamp")
    print(result.head())

    assert result.to_dicts() == [
        {
            "series_id": "A",
            "cutoff": cutoff,
            "timestamp": datetime(2026, 1, 2),
            "y_pred": 10.0,
            "y_true": 12.0,
            "is_evaluable": True,
        },
        {
            "series_id": "A",
            "cutoff": cutoff,
            "timestamp": datetime(2026, 1, 3),
            "y_pred": 11.0,
            "y_true": None,
            "is_evaluable": False,
        },
    ]


def test_attach_actuals_marks_observed_zero_as_evaluable() -> None:
    cutoff = datetime(2026, 1, 1)
    forecasts = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "timestamp": [datetime(2026, 1, 2)],
            "y_pred": [10.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 2)],
            "y": [0.0],
        }
    )

    result = attach_actuals_to_forecasts(forecasts, actuals)

    assert result["y_true"].item() == 0.0
    assert result["is_evaluable"].item() is True


def test_attach_actuals_uses_series_and_keeps_multiple_cutoffs() -> None:
    forecasts = pl.DataFrame(
        {
            "series_id": ["A", "A", "B"],
            "cutoff": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 1),
            ],
            "timestamp": [datetime(2026, 1, 3)] * 3,
            "y_pred": [10.0, 11.0, 20.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "series_id": ["A", "B"],
            "timestamp": [datetime(2026, 1, 3)] * 2,
            "y": [12.0, 22.0],
        }
    )

    result = attach_actuals_to_forecasts(forecasts, actuals)

    assert result.height == 3
    assert result.sort("series_id", "cutoff").select(
        "series_id", "cutoff", "y_true", "is_evaluable"
    ).rows() == [
        ("A", datetime(2026, 1, 1), 12.0, True),
        ("A", datetime(2026, 1, 2), 12.0, True),
        ("B", datetime(2026, 1, 1), 22.0, True),
    ]


def test_attach_actuals_rejects_duplicate_observations() -> None:
    forecasts = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [datetime(2026, 1, 1)],
            "timestamp": [datetime(2026, 1, 2)],
            "y_pred": [10.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "timestamp": [datetime(2026, 1, 2), datetime(2026, 1, 2)],
            "y": [12.0, 15.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate observations"):
        attach_actuals_to_forecasts(forecasts, actuals)


def test_attach_actuals_rejects_duplicates_outside_forecasts() -> None:
    forecasts = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [datetime(2026, 1, 1)],
            "timestamp": [datetime(2026, 1, 2)],
            "y_pred": [10.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "series_id": ["A", "B", "B"],
            "timestamp": [datetime(2026, 1, 2)] * 3,
            "y": [12.0, 15.0, 16.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate observations"):
        attach_actuals_to_forecasts(forecasts, actuals)


def test_evaluate_wape_uses_only_evaluable_days() -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A", "A"],
            "cutoff": [cutoff] * 3,
            "timestamp": [
                datetime(2026, 1, 2),
                datetime(2026, 1, 3),
                datetime(2026, 1, 4),
            ],
            "y_pred": [10.0, 10.0, 100.0],
            "y_true": [12.0, 8.0, None],
            "is_evaluable": [True, True, False],
        }
    )

    scores = evaluate_wape_by_series(evaluation)

    assert scores.height == 1
    assert scores["series_id"].item() == "A"
    assert scores["cutoff"].item() == cutoff
    assert scores["n_forecasts"].item() == 3
    assert scores["n_evaluable"].item() == 2
    assert scores["wape"].item() == pytest.approx(0.2)
    assert scores["wape_status"].item() == "ok"


@pytest.mark.parametrize(
    ("actual_values", "evaluable_flags", "expected_count", "expected_status"),
    [
        pytest.param(
            [None, None],
            [False, False],
            0,
            "no_evaluable_days",
            id="no-evaluable-days",
        ),
        pytest.param(
            [0.0, 0.0],
            [True, True],
            2,
            "zero_actual_sum",
            id="zero-actual-sum",
        ),
    ],
)
def test_evaluate_wape_reports_undefined_scores(
    actual_values: list[float | None],
    evaluable_flags: list[bool],
    expected_count: int,
    expected_status: str,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [cutoff, cutoff],
            "timestamp": [datetime(2026, 1, 2), datetime(2026, 1, 3)],
            "y_pred": [5.0, 8.0],
            "y_true": actual_values,
            "is_evaluable": evaluable_flags,
        },
        schema_overrides={"y_true": pl.Float64},
    )

    scores = evaluate_wape_by_series(evaluation)

    assert scores.height == 1
    assert scores["n_forecasts"].item() == 2
    assert scores["n_evaluable"].item() == expected_count
    assert scores["wape"].item() is None
    assert scores["wape_status"].item() == expected_status
    assert scores.schema["wape"] == pl.Float64


@pytest.mark.parametrize("y_pred", [None, np.nan, np.inf, -np.inf])
def test_evaluate_wape_rejects_invalid_prediction_when_actuals_are_zero(
    y_pred: float | None,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [0.0],
            "y_pred": [y_pred],
            "is_evaluable": [True],
        },
        schema_overrides={"y_pred": pl.Float64},
    )

    with pytest.raises(ValueError, match="y_pred must not contain null or non-finite values"):
        evaluate_wape_by_series(evaluation)


@pytest.mark.parametrize("y_true", [None, np.nan, np.inf, -np.inf])
def test_evaluate_wape_rejects_invalid_evaluable_actual(y_true: float | None) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [y_true],
            "y_pred": [1.0],
            "is_evaluable": [True],
        },
        schema_overrides={"y_true": pl.Float64},
    )

    with pytest.raises(ValueError, match="y_true must not contain null or non-finite values"):
        evaluate_wape_by_series(evaluation)


def test_evaluate_wape_ignores_invalid_values_on_unevaluable_days() -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [cutoff, cutoff],
            "y_true": [10.0, None],
            "y_pred": [8.0, np.nan],
            "is_evaluable": [True, False],
        }
    )

    scores = evaluate_wape_by_series(evaluation)

    assert scores["n_forecasts"].item() == 2
    assert scores["n_evaluable"].item() == 1
    assert scores["wape"].item() == pytest.approx(0.2)
    assert scores["wape_status"].item() == "ok"


def test_evaluate_wape_separates_series_and_cutoffs() -> None:
    first_cutoff = datetime(2026, 1, 1)
    second_cutoff = datetime(2026, 1, 2)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A", "B"],
            "cutoff": [first_cutoff, second_cutoff, first_cutoff],
            "timestamp": [datetime(2026, 1, 3)] * 3,
            "y_pred": [8.0, 5.0, 3.0],
            "y_true": [10.0, 10.0, 0.0],
            "is_evaluable": [True, True, True],
        }
    )

    scores = evaluate_wape_by_series(evaluation).sort("series_id", "cutoff")

    assert scores.height == 3
    assert scores["series_id"].to_list() == ["A", "A", "B"]
    assert scores["cutoff"].to_list() == [
        first_cutoff,
        second_cutoff,
        first_cutoff,
    ]
    assert scores["wape"][0] == pytest.approx(0.2)
    assert scores["wape"][1] == pytest.approx(0.5)
    assert scores["wape"][2] is None
    assert scores["wape_status"].to_list() == ["ok", "ok", "zero_actual_sum"]


def test_evaluate_wape_with_no_forecasts() -> None:
    evaluation = pl.DataFrame(
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "timestamp": pl.Datetime("us"),
            "y_pred": pl.Float64,
            "y_true": pl.Float64,
            "is_evaluable": pl.Boolean,
        }
    )

    scores = evaluate_wape_by_series(evaluation)

    assert scores.is_empty()
    assert scores.schema == {
        "series_id": pl.String,
        "cutoff": pl.Datetime("us"),
        "n_forecasts": pl.Int64,
        "n_evaluable": pl.Int64,
        "wape": pl.Float64,
        "wape_status": pl.String,
    }


def test_backtest_forecasts_can_be_evaluated() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 182,
            "timestamp": [start + timedelta(days=day) for day in range(182)],
            "y": [5.0] * 180 + [10.0, 20.0],
        }
    )

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            pass

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, 5.0, dtype=np.float64)

    reports, forecasts, references = run_backtest(data, [cutoff], FakeModel, horizon=3)
    evaluation = attach_actuals_to_forecasts(forecasts, data)
    scores = evaluate_wape_by_series(evaluation)

    assert reports["eligible"].item() is True
    assert scores["n_forecasts"].item() == 3
    assert scores["n_evaluable"].item() == 2
    assert scores["wape"].item() == pytest.approx(2 / 3)
    assert scores["wape_status"].item() == "ok"


def test_evaluate_scaled_metrics_by_series() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A", "A"],
            "cutoff": [cutoff] * 3,
            "y_true": [10.0, 12.0, None],
            "y_pred": [9.0, 15.0, 8.0],
            "is_evaluable": [True, True, False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores.height == 1
    assert scores["series_id"].item() == "A"
    assert scores["cutoff"].item() == cutoff
    assert scores["season_length"].item() == 7
    assert scores["n_forecasts"].item() == 3
    assert scores["n_evaluable"].item() == 2
    assert scores["mase"].item() == pytest.approx(1.0)
    assert scores["rmsse"].item() == pytest.approx(np.sqrt(5 / 7))
    assert scores["mase_status"].item() == "ok"
    assert scores["rmsse_status"].item() == "ok"


def test_evaluate_scaled_metrics_with_zero_reference() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [10.0],
            "y_pred": [9.0],
            "is_evaluable": [True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [0.0],
            "naive_mse": [0.0],
            "reference_status": ["zero_reference"],
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores.height == 1
    assert scores["n_forecasts"].item() == 1
    assert scores["n_evaluable"].item() == 1
    assert scores["mase"].item() is None
    assert scores["mase_status"].item() == "zero_reference"
    assert scores["rmsse"].item() is None
    assert scores["rmsse_status"].item() == "zero_reference"


def test_evaluate_scaled_metrics_without_evaluable_days() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [cutoff, cutoff],
            "y_true": [None, None],
            "y_pred": [9.0, 10.0],
            "is_evaluable": [False, False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores.height == 1
    assert scores["n_forecasts"].item() == 2
    assert scores["n_evaluable"].item() == 0
    assert scores["mase"].item() is None
    assert scores["mase_status"].item() == "no_evaluable_days"
    assert scores["rmsse"].item() is None
    assert scores["rmsse_status"].item() == "no_evaluable_days"


def test_evaluate_scaled_metrics_rejects_missing_reference() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [10.0],
            "y_pred": [9.0],
            "is_evaluable": [True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["B"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        }
    )

    with pytest.raises(ValueError, match="reference is missing"):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_evaluate_scaled_metrics_rejects_duplicate_references() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [10.0],
            "y_pred": [9.0],
            "is_evaluable": [True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [cutoff, cutoff],
            "season_length": [7, 7],
            "naive_mae": [2.0, 3.0],
            "naive_mse": [7.0, 9.0],
            "reference_status": ["ok", "ok"],
        }
    )

    with pytest.raises(ValueError, match="unique reference"):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_evaluate_scaled_metrics_without_forecasts() -> None:
    evaluation = pl.DataFrame(
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "y_true": pl.Float64,
            "y_pred": pl.Float64,
            "is_evaluable": pl.Boolean,
        }
    )
    references = pl.DataFrame(
        schema={
            "series_id": pl.String,
            "cutoff": pl.Datetime("us"),
            "season_length": pl.Int64,
            "naive_mae": pl.Float64,
            "naive_mse": pl.Float64,
            "reference_status": pl.String,
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores.is_empty()
    assert scores.schema == {
        "series_id": pl.String,
        "cutoff": pl.Datetime("us"),
        "season_length": pl.Int64,
        "n_forecasts": pl.Int64,
        "n_evaluable": pl.Int64,
        "mase": pl.Float64,
        "mase_status": pl.String,
        "rmsse": pl.Float64,
        "rmsse_status": pl.String,
    }


@pytest.mark.parametrize(
    ("naive_mae", "naive_mse"),
    [(-1.0, 7.0), (2.0, -1.0), (np.nan, 7.0), (2.0, np.inf)],
)
def test_evaluate_scaled_metrics_rejects_invalid_reference_without_evaluable_days(
    naive_mae: float,
    naive_mse: float,
) -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [naive_mae],
            "naive_mse": [naive_mse],
            "reference_status": ["ok"],
        }
    )

    with pytest.raises(ValueError):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_evaluate_scaled_metrics_rejects_nonfinite_prediction_with_zero_reference() -> None:
    cutoff = datetime(2026, 1, 1)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [10.0],
            "y_pred": [np.nan],
            "is_evaluable": [True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [0.0],
            "naive_mse": [0.0],
            "reference_status": ["zero_reference"],
        }
    )

    with pytest.raises(ValueError):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_evaluate_scaled_metrics_matches_series_and_cutoff() -> None:
    cutoff_1 = datetime(2026, 1, 1)
    cutoff_2 = datetime(2026, 1, 2)

    evaluation = pl.DataFrame(
        {
            "series_id": ["A", "A", "B"],
            "cutoff": [cutoff_1, cutoff_2, cutoff_1],
            "y_true": [10.0, 10.0, 10.0],
            "y_pred": [8.0, 8.0, 8.0],
            "is_evaluable": [True, True, True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["B", "A", "A"],
            "cutoff": [cutoff_1, cutoff_2, cutoff_1],
            "season_length": [7, 7, 7],
            "naive_mae": [1.0, 4.0, 2.0],
            "naive_mse": [1.0, 16.0, 4.0],
            "reference_status": ["ok", "ok", "ok"],
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)
    rows = {(row["series_id"], row["cutoff"]): row for row in scores.to_dicts()}

    assert len(rows) == 3
    assert rows[("A", cutoff_1)]["mase"] == pytest.approx(1.0)
    assert rows[("A", cutoff_1)]["rmsse"] == pytest.approx(1.0)
    assert rows[("A", cutoff_2)]["mase"] == pytest.approx(0.5)
    assert rows[("A", cutoff_2)]["rmsse"] == pytest.approx(0.5)
    assert rows[("B", cutoff_1)]["mase"] == pytest.approx(2.0)
    assert rows[("B", cutoff_1)]["rmsse"] == pytest.approx(2.0)


def test_backtest_to_scaled_metrics() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 182,
            "timestamp": [start + timedelta(days=day) for day in range(182)],
            "y": [float(day + 1) for day in range(182)],
        }
    )

    class LastValueModel:
        def __init__(self) -> None:
            self.last_value = 0.0

        def fit(self, train: NDArray[np.float64]) -> None:
            self.last_value = float(train[-1])

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, self.last_value, dtype=np.float64)

    _, forecasts, references = run_backtest(
        data,
        [cutoff],
        LastValueModel,
        horizon=2,
        season_length=7,
    )
    evaluation = attach_actuals_to_forecasts(forecasts, data)
    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores.height == 1
    assert scores["n_forecasts"].item() == 2
    assert scores["n_evaluable"].item() == 2
    assert scores["mase"].item() == pytest.approx(1.5 / 7)
    assert scores["rmsse"].item() == pytest.approx(np.sqrt(2.5 / 49))
    assert scores["mase_status"].item() == "ok"
    assert scores["rmsse_status"].item() == "ok"


@pytest.mark.parametrize(
    ("naive_mae", "naive_mse"),
    [(None, 7.0), (2.0, None)],
)
def test_evaluate_scaled_metrics_rejects_null_reference(
    naive_mae: float | None,
    naive_mse: float | None,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [naive_mae],
            "naive_mse": [naive_mse],
            "reference_status": ["ok"],
        }
    )

    with pytest.raises(ValueError):
        evaluate_scaled_metrics_by_series(evaluation, references)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [(np.inf, 9.0), (None, 9.0), (10.0, None)],
)
def test_evaluate_scaled_metrics_rejects_invalid_evaluable_values(
    y_true: float | None,
    y_pred: float | None,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [y_true],
            "y_pred": [y_pred],
            "is_evaluable": [True],
        },
        schema_overrides={"y_true": pl.Float64, "y_pred": pl.Float64},
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [0.0],
            "naive_mse": [0.0],
            "reference_status": ["zero_reference"],
        }
    )

    with pytest.raises(ValueError):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_evaluate_scaled_metrics_no_days_with_zero_reference() -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [0.0],
            "naive_mse": [0.0],
            "reference_status": ["zero_reference"],
        }
    )

    scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert scores["mase"].item() is None
    assert scores["rmsse"].item() is None
    assert scores["mase_status"].item() == "no_evaluable_days"
    assert scores["rmsse_status"].item() == "no_evaluable_days"


def test_evaluate_scaled_metrics_rejects_wrong_cutoff_reference() -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [10.0],
            "y_pred": [9.0],
            "is_evaluable": [True],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff + timedelta(days=1)],
            "season_length": [7],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        }
    )

    with pytest.raises(ValueError, match="reference is missing"):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_run_backtest_rejects_duplicate_cutoffs() -> None:
    cutoff = datetime(2026, 1, 1)
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [cutoff],
            "y": [5.0],
        }
    )

    def make_model() -> ForecastModel:
        raise AssertionError("A model should not be created")

    with pytest.raises(ValueError, match="cutoffs must be unique"):
        run_backtest(data, [cutoff, cutoff], make_model, horizon=2)


@pytest.mark.parametrize("season_length", [0, -1, None])
def test_evaluate_scaled_metrics_rejects_invalid_season_length(
    season_length: int | None,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [season_length],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        },
        schema_overrides={"season_length": pl.Int64},
    )

    with pytest.raises(ValueError, match="season_length"):
        evaluate_scaled_metrics_by_series(evaluation, references)


@pytest.mark.parametrize("season_length", [True, False, 3.0, 3.5, "3"])
def test_evaluate_scaled_metrics_rejects_non_integer_season_length(
    season_length: bool | float | str,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [season_length],
            "naive_mae": [2.0],
            "naive_mse": [7.0],
            "reference_status": ["ok"],
        }
    )

    with pytest.raises(ValueError, match="season_length must be a positive integer"):
        evaluate_scaled_metrics_by_series(evaluation, references)


@pytest.mark.parametrize(
    ("naive_mae", "naive_mse", "reference_status"),
    [
        (0.0, 0.0, "ok"),
        (2.0, 7.0, "zero_reference"),
        (2.0, 7.0, "unknown"),
        (2.0, 7.0, None),
    ],
)
def test_evaluate_scaled_metrics_rejects_invalid_reference_status(
    naive_mae: float,
    naive_mse: float,
    reference_status: str | None,
) -> None:
    cutoff = datetime(2026, 1, 1)
    evaluation = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "y_true": [None],
            "y_pred": [9.0],
            "is_evaluable": [False],
        }
    )
    references = pl.DataFrame(
        {
            "series_id": ["A"],
            "cutoff": [cutoff],
            "season_length": [7],
            "naive_mae": [naive_mae],
            "naive_mse": [naive_mse],
            "reference_status": [reference_status],
        },
        schema_overrides={"reference_status": pl.String},
    )

    with pytest.raises(ValueError, match="reference_status"):
        evaluate_scaled_metrics_by_series(evaluation, references)


def test_prepare_training_at_cutoff_rejects_duplicate_observations() -> None:
    cutoff = datetime(2026, 1, 1)
    data = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "timestamp": [cutoff, cutoff],
            "y": [5.0, 6.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate observations"):
        prepare_training_at_cutoff(data, cutoff)


def test_prepare_training_at_cutoff_ignores_future_duplicates() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)
    future = cutoff + timedelta(days=1)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 182,
            "timestamp": [start + timedelta(days=day) for day in range(180)] + [future, future],
            "y": [5.0] * 180 + [6.0, 7.0],
        }
    )

    report, train = prepare_training_at_cutoff(data, cutoff)

    assert report["eligible"].item() is True
    assert train.height == 180
    assert train["timestamp"].max() == cutoff


def test_prepare_training_at_cutoff_allows_same_timestamp_for_distinct_series() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 180 + ["B"] * 180,
            "timestamp": [start + timedelta(days=day) for day in range(180)] * 2,
            "y": [5.0] * 180 + [10.0] * 180,
        }
    )

    report, train = prepare_training_at_cutoff(data, cutoff)

    assert report["eligible"].sum() == 2
    assert train.height == 360
    assert set(train["series_id"]) == {"A", "B"}
