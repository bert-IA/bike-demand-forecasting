from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandera
import polars as pl
import pytest
from numpy.typing import NDArray

from forecasting.evaluation.backtest import (
    attach_actuals_to_forecasts,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
)
from forecasting.evaluation.benchmark import (
    build_cutoffs,
    cutoffs_to_datetimes,
    evaluate_validation_forecasts,
    get_config_date,
    get_data_input_path,
    get_experiment_name,
    get_positive_int,
    get_validation_cutoffs,
    get_validation_data,
    get_validation_section,
    load_benchmark_config,
    load_benchmark_data,
    parse_benchmark_config,
    parse_validation_config,
    run_benchmark,
    run_model_benchmark,
    run_models_benchmark,
    run_validation_backtest,
    validate_comparison_perimeter,
    validate_comparison_references,
    validate_validation_window,
)
from forecasting.models.exponential_smoothing import SeasonalExponentialSmoothing
from forecasting.models.holt_winters import DampedHoltWinters
from forecasting.models.last_value import LastValueNaive
from forecasting.models.protocol import ForecastModel
from forecasting.models.seasonal_naive import SeasonalNaive


def test_load_benchmark_config_reads_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        "experiment:\n  name: bike-validation-14d\n",
        encoding="utf-8",
    )

    result = load_benchmark_config(config_file)

    assert result == {"experiment": {"name": "bike-validation-14d"}}


@pytest.mark.parametrize("content", ["", "- first\n- second\n", "42\n"])
def test_load_benchmark_config_rejects_non_mapping(
    tmp_path: Path,
    content: str,
) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_benchmark_config(config_file)


def test_load_benchmark_config_rejects_non_string_root_key(tmp_path: Path) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text("42: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys must be strings"):
        load_benchmark_config(config_file)


def test_get_validation_section_returns_mapping() -> None:
    config: dict[str, object] = {"validation": {"horizon_days": 14}}

    result = get_validation_section(config)

    assert result == {"horizon_days": 14}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"validation": None},
        {"validation": 14},
        {"validation": ["not", "a", "mapping"]},
        {"validation": {42: "invalid key"}},
    ],
)
def test_get_validation_section_rejects_invalid_mapping(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="validation"):
        get_validation_section(config)


@pytest.mark.parametrize("name", ["horizon_days", "step_days"])
def test_get_positive_int_accepts_positive_integer(name: str) -> None:
    section: dict[str, object] = {name: 14}

    assert get_positive_int(section, name) == 14


@pytest.mark.parametrize("value", [0, -1, 2.5, "14", True, False, None])
def test_get_positive_int_rejects_invalid_value(value: object) -> None:
    section: dict[str, object] = {"horizon_days": value}

    with pytest.raises(ValueError, match="horizon_days"):
        get_positive_int(section, "horizon_days")


def test_get_positive_int_rejects_missing_parameter() -> None:
    section: dict[str, object] = {}

    with pytest.raises(ValueError, match="horizon_days"):
        get_positive_int(section, "horizon_days")


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("first_cutoff", "2026-03-23", date(2026, 3, 23)),
        ("last_cutoff", "2026-09-07", date(2026, 9, 7)),
        ("test_start", "2026-09-22", date(2026, 9, 22)),
    ],
)
def test_get_config_date_accepts_iso_date(
    name: str,
    text: str,
    expected: date,
) -> None:
    section: dict[str, object] = {name: text}

    assert get_config_date(section, name) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        20260323,
        True,
        date(2026, 3, 23),
        "",
        "23-03-2026",
        "20260323",
        "2026-3-23",
        "2026-02-30",
        "2026-03-23T12:00:00",
    ],
)
def test_get_config_date_rejects_invalid_value(value: object) -> None:
    section: dict[str, object] = {"first_cutoff": value}

    with pytest.raises(ValueError, match="first_cutoff"):
        get_config_date(section, "first_cutoff")


def test_get_config_date_rejects_missing_parameter() -> None:
    section: dict[str, object] = {}

    with pytest.raises(ValueError, match="first_cutoff"):
        get_config_date(section, "first_cutoff")


def test_validate_validation_window_accepts_safe_window() -> None:
    validate_validation_window(
        first_cutoff=date(2026, 3, 23),
        last_cutoff=date(2026, 9, 7),
        horizon_days=14,
        test_start=date(2026, 9, 22),
    )


def test_validate_validation_window_rejects_reversed_cutoffs() -> None:
    with pytest.raises(ValueError, match="first_cutoff"):
        validate_validation_window(
            first_cutoff=date(2026, 9, 7),
            last_cutoff=date(2026, 3, 23),
            horizon_days=14,
            test_start=date(2026, 9, 22),
        )


@pytest.mark.parametrize(
    "test_start",
    [date(2026, 9, 21), date(2026, 9, 20)],
)
def test_validate_validation_window_rejects_overlap_with_test(
    test_start: date,
) -> None:
    with pytest.raises(ValueError, match="test"):
        validate_validation_window(
            first_cutoff=date(2026, 3, 23),
            last_cutoff=date(2026, 9, 7),
            horizon_days=14,
            test_start=test_start,
        )


def test_build_cutoffs_returns_regular_dates() -> None:
    result = build_cutoffs(
        first_cutoff=date(2026, 3, 1),
        last_cutoff=date(2026, 3, 15),
        step_days=7,
    )

    assert result == [
        date(2026, 3, 1),
        date(2026, 3, 8),
        date(2026, 3, 15),
    ]


def test_build_cutoffs_matches_bike_validation_dates() -> None:
    result = build_cutoffs(
        first_cutoff=date(2026, 3, 23),
        last_cutoff=date(2026, 9, 7),
        step_days=14,
    )

    assert len(result) == 13
    assert result[0] == date(2026, 3, 23)
    assert result[1] == date(2026, 4, 6)
    assert result[-1] == date(2026, 9, 7)


def test_build_cutoffs_accepts_a_single_cutoff() -> None:
    result = build_cutoffs(
        first_cutoff=date(2026, 3, 23),
        last_cutoff=date(2026, 3, 23),
        step_days=14,
    )

    assert result == [date(2026, 3, 23)]


def test_build_cutoffs_rejects_reversed_dates() -> None:
    with pytest.raises(ValueError, match="first_cutoff"):
        build_cutoffs(
            first_cutoff=date(2026, 9, 7),
            last_cutoff=date(2026, 3, 23),
            step_days=14,
        )


def test_build_cutoffs_rejects_unaligned_last_cutoff() -> None:
    with pytest.raises(ValueError, match="last_cutoff"):
        build_cutoffs(
            first_cutoff=date(2026, 3, 1),
            last_cutoff=date(2026, 3, 20),
            step_days=7,
        )


@pytest.mark.parametrize("step_days", [0, -1, True, False])
def test_build_cutoffs_rejects_invalid_step(step_days: int) -> None:
    with pytest.raises(ValueError, match="step_days"):
        build_cutoffs(
            first_cutoff=date(2026, 3, 1),
            last_cutoff=date(2026, 3, 15),
            step_days=step_days,
        )


def test_build_cutoffs_rejects_float_step() -> None:
    with pytest.raises(ValueError, match="step_days"):
        build_cutoffs(
            first_cutoff=date(2026, 3, 1),
            last_cutoff=date(2026, 3, 15),
            step_days=2.5,  # type: ignore[arg-type]
        )


def test_build_cutoffs_accepts_maximum_date_as_last_cutoff() -> None:
    result = build_cutoffs(
        first_cutoff=date(9999, 12, 30),
        last_cutoff=date.max,
        step_days=1,
    )

    assert result == [date(9999, 12, 30), date.max]


def test_get_validation_cutoffs_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "benchmark.yaml"
    config_file.write_text(
        "validation:\n"
        '  train_start: "2025-09-25"\n'
        "  initial_train_days: 180\n"
        '  first_cutoff: "2026-03-23"\n'
        '  last_cutoff: "2026-09-07"\n'
        "  horizon_days: 14\n"
        "  step_days: 14\n"
        '  test_start: "2026-09-22"\n',
        encoding="utf-8",
    )

    config = load_benchmark_config(config_file)
    result = get_validation_cutoffs(config)
    assert len(result) == 13
    assert result[0] == date(2026, 3, 23)
    assert result[-1] == date(2026, 9, 7)


def test_get_validation_cutoffs_rejects_missing_section() -> None:
    config: dict[str, object] = {}

    with pytest.raises(ValueError, match="validation"):
        get_validation_cutoffs(config)


@pytest.mark.parametrize(
    ("name", "value", "error_message"),
    [
        ("train_start", "2026-03-24", "train_start"),
        ("initial_train_days", 0, "initial_train_days"),
        ("first_cutoff", "2026-02-30", "first_cutoff"),
        ("horizon_days", 0, "horizon_days"),
        ("step_days", 0, "step_days"),
        ("last_cutoff", "2026-03-01", "first_cutoff"),
        ("last_cutoff", "2026-09-06", "last_cutoff"),
        ("test_start", "2026-09-21", "test_start"),
    ],
)
def test_get_validation_cutoffs_rejects_invalid_config(
    name: str,
    value: object,
    error_message: str,
) -> None:
    section: dict[str, object] = {
        "train_start": "2025-09-25",
        "initial_train_days": 180,
        "first_cutoff": "2026-03-23",
        "last_cutoff": "2026-09-07",
        "horizon_days": 14,
        "step_days": 14,
        "test_start": "2026-09-22",
    }
    section[name] = value
    config: dict[str, object] = {"validation": section}

    with pytest.raises(ValueError, match=error_message):
        get_validation_cutoffs(config)


def test_parse_validation_config_returns_validated_parameters() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2025-09-25",
            "initial_train_days": 180,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-09-07",
            "horizon_days": 14,
            "step_days": 14,
            "test_start": "2026-09-22",
        }
    }

    result = parse_validation_config(config)

    assert result.train_start == date(2025, 9, 25)
    assert result.initial_train_days == 180
    assert result.first_cutoff == date(2026, 3, 23)
    assert result.last_cutoff == date(2026, 9, 7)
    assert result.horizon_days == 14
    assert result.step_days == 14
    assert result.test_start == date(2026, 9, 22)
    assert len(result.cutoffs) == 13
    assert result.cutoffs[0] == result.first_cutoff
    assert result.cutoffs[-1] == result.last_cutoff


def test_parse_validation_config_rejects_overlap_with_test() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2025-09-25",
            "initial_train_days": 180,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-09-07",
            "horizon_days": 14,
            "step_days": 14,
            "test_start": "2026-09-21",
        }
    }

    with pytest.raises(ValueError, match="test_start"):
        parse_validation_config(config)


def test_cutoffs_to_datetimes_returns_midnight_datetimes() -> None:
    cutoffs = (date(2026, 3, 23), date(2026, 4, 6))

    result = cutoffs_to_datetimes(cutoffs)

    assert result == [
        datetime(2026, 3, 23, 0, 0),
        datetime(2026, 4, 6, 0, 0),
    ]
    assert all(value.tzinfo is None for value in result)


def test_cutoffs_to_datetimes_accepts_empty_tuple() -> None:
    assert cutoffs_to_datetimes(()) == []


def test_get_experiment_name_returns_name() -> None:
    config: dict[str, object] = {"experiment": {"name": "bike-validation-14d"}}

    assert get_experiment_name(config) == "bike-validation-14d"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"experiment": None},
        {"experiment": "not a mapping"},
        {"experiment": {}},
        {"experiment": {"name": 42}},
        {"experiment": {"name": ""}},
        {"experiment": {"name": "   "}},
    ],
)
def test_get_experiment_name_rejects_invalid_config(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="experiment"):
        get_experiment_name(config)


def test_get_data_input_path_returns_path() -> None:
    config: dict[str, object] = {"data": {"input_path": "data/processed/bike_daily.parquet"}}

    assert get_data_input_path(config) == "data/processed/bike_daily.parquet"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"data": None},
        {"data": "not a mapping"},
        {"data": {}},
        {"data": {"input_path": 42}},
        {"data": {"input_path": ""}},
        {"data": {"input_path": "   "}},
    ],
)
def test_get_data_input_path_rejects_invalid_config(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="data"):
        get_data_input_path(config)


def test_parse_benchmark_config_returns_settings() -> None:
    config: dict[str, object] = {
        "experiment": {"name": "bike-validation-14d"},
        "data": {"input_path": "data/processed/bike_daily.parquet"},
        "validation": {
            "train_start": "2025-09-25",
            "initial_train_days": 180,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-09-07",
            "horizon_days": 14,
            "step_days": 14,
            "test_start": "2026-09-22",
        },
    }

    result = parse_benchmark_config(config)

    assert result.experiment_name == "bike-validation-14d"
    assert result.input_path == "data/processed/bike_daily.parquet"
    assert result.validation.horizon_days == 14
    assert len(result.validation.cutoffs) == 13


def test_parse_benchmark_config_rejects_missing_data() -> None:
    config: dict[str, object] = {
        "experiment": {"name": "bike-validation-14d"},
        "validation": {
            "train_start": "2025-09-25",
            "initial_train_days": 180,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-09-07",
            "horizon_days": 14,
            "step_days": 14,
            "test_start": "2026-09-22",
        },
    }

    with pytest.raises(ValueError, match="data"):
        parse_benchmark_config(config)


def test_load_benchmark_data_reads_relative_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    expected = pl.DataFrame(
        {
            "series_id": ["counter-1"],
            "timestamp": [datetime(2026, 3, 23)],
            "y": [42.0],
        }
    )
    expected.write_parquet(data_dir / "bike.parquet")

    result = load_benchmark_data(tmp_path, "data/bike.parquet")

    assert result.equals(expected)


def test_load_benchmark_data_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        load_benchmark_data(tmp_path, "data/missing.parquet")


def test_load_benchmark_data_rejects_relative_project_root() -> None:
    with pytest.raises(ValueError, match="project_root"):
        load_benchmark_data(Path("relative/project"), "data/bike.parquet")


def test_load_benchmark_data_rejects_negative_counts(tmp_path: Path) -> None:
    data = pl.DataFrame(
        {
            "series_id": ["counter-1"],
            "timestamp": [datetime(2026, 3, 23)],
            "y": [-1.0],
        }
    )
    data.write_parquet(tmp_path / "invalid.parquet")

    with pytest.raises(pandera.errors.SchemaError):
        load_benchmark_data(tmp_path, "invalid.parquet")


def test_get_validation_data_respects_date_boundaries() -> None:
    data = pl.DataFrame(
        {
            "series_id": ["counter-1"] * 4,
            "timestamp": [
                datetime(2026, 9, 20),
                datetime(2026, 9, 21),
                datetime(2026, 9, 22),
                datetime(2026, 9, 23),
            ],
            "y": [10.0, 12.0, 14.0, 100.0],
        }
    )

    result = get_validation_data(
        data,
        train_start=date(2026, 9, 21),
        test_start=date(2026, 9, 23),
    )

    assert result["timestamp"].to_list() == [
        datetime(2026, 9, 21),
        datetime(2026, 9, 22),
    ]
    assert result["y"].to_list() == [12.0, 14.0]


@pytest.mark.parametrize("train_start", ["2026-03-20", "2026-03-21"])
def test_parse_validation_config_accepts_sufficient_train_window(
    train_start: str,
) -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": train_start,
            "initial_train_days": 3,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }

    result = parse_validation_config(config)

    assert result.initial_train_days == 3


def test_parse_validation_config_rejects_short_train_window() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-22",
            "initial_train_days": 3,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }

    with pytest.raises(ValueError, match="initial_train_days"):
        parse_validation_config(config)


def test_run_validation_backtest_uses_configured_training_window() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-16",
            "initial_train_days": 8,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }
    validation = parse_validation_config(config)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 11,
            "timestamp": [datetime(2026, 3, day) for day in range(15, 26)],
            "y": [99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 50.0, 100.0],
        }
    )
    trained_values: list[list[float]] = []

    class FakeModel:
        def fit(self, train: NDArray[np.float64]) -> None:
            trained_values.append(train.tolist())

        def predict(self, horizon: int) -> NDArray[np.float64]:
            return np.full(horizon, 8.0, dtype=np.float64)

    reports, forecasts, references = run_validation_backtest(data, validation, FakeModel)

    validation_data = get_validation_data(data, validation.train_start, validation.test_start)
    evaluation = attach_actuals_to_forecasts(forecasts, validation_data)

    wape_scores = evaluate_wape_by_series(evaluation)
    scaled_scores = evaluate_scaled_metrics_by_series(evaluation, references)

    assert reports["eligible"].item() is True
    assert trained_values == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    assert forecasts["timestamp"].to_list() == [datetime(2026, 3, 24)]
    assert forecasts["y_pred"].to_list() == [8.0]

    assert wape_scores["wape"].item() == pytest.approx(0.84)
    assert scaled_scores["mase"].item() == pytest.approx(6.0)


def test_run_validation_backtest_with_seasonal_naive() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-16",
            "initial_train_days": 8,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }
    validation = parse_validation_config(config)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 9,
            "timestamp": [datetime(2026, 3, day) for day in range(16, 25)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 50.0],
        }
    )

    reports, forecasts, references = run_validation_backtest(
        data,
        validation,
        lambda: SeasonalNaive(season_length=7),
    )

    evaluation, wape_scores, scaled_scores = evaluate_validation_forecasts(
        data, validation, forecasts, references
    )

    assert reports["eligible"].item() is True
    assert forecasts["y_pred"].item() == pytest.approx(2.0)
    assert evaluation["y_true"].item() == 50.0
    assert wape_scores["wape"].item() == pytest.approx(0.96)
    assert scaled_scores["mase"].item() == pytest.approx(48 / 7)

    result = run_model_benchmark(
        data,
        validation,
        lambda: SeasonalNaive(season_length=7),
    )

    assert result.forecasts["y_pred"].item() == pytest.approx(2.0)
    assert result.evaluation["y_true"].item() == 50.0
    assert result.wape_scores["wape"].item() == pytest.approx(0.96)
    assert result.scaled_scores["mase"].item() == pytest.approx(48 / 7)


def test_run_models_benchmark_returns_results_by_model() -> None:
    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-16",
            "initial_train_days": 8,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }
    validation = parse_validation_config(config)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 9,
            "timestamp": [datetime(2026, 3, day) for day in range(16, 25)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 50.0],
        }
    )

    models: dict[str, Callable[[], ForecastModel]] = {
        "seasonal_7": lambda: SeasonalNaive(season_length=7),
        "seasonal_3": lambda: SeasonalNaive(season_length=3),
        "last_value": LastValueNaive,
    }

    results = run_models_benchmark(data, validation, models)

    assert set(results) == {"seasonal_7", "seasonal_3", "last_value"}
    assert results["last_value"].forecasts["y_pred"].item() == pytest.approx(8.0)
    assert results["seasonal_7"].forecasts["y_pred"].item() == pytest.approx(2.0)
    assert results["seasonal_3"].forecasts["y_pred"].item() == pytest.approx(6.0)


def test_run_models_benchmark_with_baseline_models() -> None:
    start = datetime(2026, 1, 1)
    cutoff = start + timedelta(days=179)

    config: dict[str, object] = {
        "validation": {
            "train_start": start.date().isoformat(),
            "initial_train_days": 180,
            "first_cutoff": cutoff.date().isoformat(),
            "last_cutoff": cutoff.date().isoformat(),
            "horizon_days": 1,
            "step_days": 1,
            "test_start": (cutoff + timedelta(days=2)).date().isoformat(),
        }
    }
    validation = parse_validation_config(config)

    data = pl.DataFrame(
        {
            "series_id": ["A"] * 181,
            "timestamp": [start + timedelta(days=day) for day in range(181)],
            "y": [100.0 + (day % 7) * 10.0 + day * 0.01 for day in range(181)],
        }
    )

    models: dict[str, Callable[[], ForecastModel]] = {
        "last_value": LastValueNaive,
        "seasonal_naive": lambda: SeasonalNaive(season_length=7),
        "exponential_smoothing": lambda: SeasonalExponentialSmoothing(season_length=7),
        "holt_winters_damped": lambda: DampedHoltWinters(season_length=7),
    }

    results = run_models_benchmark(data, validation, models)

    assert set(results) == set(models)
    for result in results.values():
        assert result.reports["eligible"].item() is True
        assert result.forecasts.height == 1
        assert result.forecasts["y_pred"].is_finite().all()
        assert result.scaled_scores["mase_status"].item() == "ok"


def test_run_benchmark_loads_config_and_data(tmp_path: Path) -> None:
    data = pl.DataFrame(
        {
            "series_id": ["A"] * 9,
            "timestamp": [datetime(2026, 3, day) for day in range(16, 25)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 50.0],
        }
    )
    data.write_parquet(tmp_path / "data.parquet")

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "experiment:\n"
        "  name: test-benchmark\n"
        "data:\n"
        "  input_path: data.parquet\n"
        "validation:\n"
        '  train_start: "2026-03-16"\n'
        "  initial_train_days: 8\n"
        '  first_cutoff: "2026-03-23"\n'
        '  last_cutoff: "2026-03-23"\n'
        "  horizon_days: 1\n"
        "  step_days: 1\n"
        '  test_start: "2026-03-25"\n',
        encoding="utf-8",
    )

    config_load = load_benchmark_config(config_path)
    config = parse_benchmark_config(config_load)

    data = load_benchmark_data(tmp_path, config.input_path)

    models: dict[str, Callable[[], ForecastModel]] = {
        "last_value": LastValueNaive,
    }

    config_returned, results = run_benchmark(config, data, models)

    assert config_returned.experiment_name == "test-benchmark"
    assert results["last_value"].wape_scores["wape"].item() == pytest.approx(0.84)
    assert results["last_value"].forecasts["y_pred"].item() == pytest.approx(8.0)


@pytest.fixture
def comparison_evaluation() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "cutoff": [datetime(2026, 3, 23)] * 2,
            "timestamp": [datetime(2026, 3, 24), datetime(2026, 3, 25)],
            "y_true": [10.0, None],
            "is_evaluable": [True, False],
            "y_pred": [9.0, 9.0],
        }
    )


def test_comparison_accepts_different_order_and_predictions(
    comparison_evaluation: pl.DataFrame,
) -> None:
    other = comparison_evaluation.reverse().with_columns(pl.lit(42.0).alias("y_pred"))

    validate_comparison_perimeter({"first": comparison_evaluation, "second": other})


def test_comparison_rejects_different_forecast_dates(
    comparison_evaluation: pl.DataFrame,
) -> None:
    other = comparison_evaluation.with_columns(
        (pl.col("timestamp") + pl.duration(days=1)).alias("timestamp")
    )

    with pytest.raises(ValueError):
        validate_comparison_perimeter({"first": comparison_evaluation, "second": other})


def test_comparison_rejects_different_actuals(
    comparison_evaluation: pl.DataFrame,
) -> None:
    other = comparison_evaluation.with_columns(pl.lit(42.0).alias("y_true"))

    with pytest.raises(ValueError):
        validate_comparison_perimeter({"first": comparison_evaluation, "second": other})


def test_comparison_rejects_duplicate_keys(
    comparison_evaluation: pl.DataFrame,
) -> None:
    duplicate = pl.concat([comparison_evaluation, comparison_evaluation.head(1)])

    with pytest.raises(ValueError):
        validate_comparison_perimeter({"first": duplicate, "second": duplicate})


def test_run_models_benchmark_rejects_different_perimeters(
    monkeypatch: pytest.MonkeyPatch,
    comparison_evaluation: pl.DataFrame,
) -> None:
    import forecasting.evaluation.benchmark as benchmark_module

    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-16",
            "initial_train_days": 8,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }
    validation = parse_validation_config(config)

    different_evaluation = comparison_evaluation.with_columns(
        (pl.col("timestamp") + pl.duration(days=1)).alias("timestamp")
    )

    def make_result(evaluation: pl.DataFrame) -> benchmark_module.ModelBenchmarkResult:
        return benchmark_module.ModelBenchmarkResult(
            reports=pl.DataFrame(),
            forecasts=pl.DataFrame(),
            references=pl.DataFrame(),
            evaluation=evaluation,
            wape_scores=pl.DataFrame(),
            scaled_scores=pl.DataFrame(),
        )

    fake_results = iter(
        [
            make_result(comparison_evaluation),
            make_result(different_evaluation),
        ]
    )

    def fake_run_model_benchmark(
        data: pl.DataFrame,
        validation: benchmark_module.ValidationConfig,
        model_factory: Callable[[], ForecastModel],
    ) -> benchmark_module.ModelBenchmarkResult:
        return next(fake_results)

    monkeypatch.setattr(
        benchmark_module,
        "run_model_benchmark",
        fake_run_model_benchmark,
    )

    models: dict[str, Callable[[], ForecastModel]] = {
        "first": LastValueNaive,
        "second": LastValueNaive,
    }

    with pytest.raises(ValueError, match="Evaluation perimeter differs.*second"):
        run_models_benchmark(pl.DataFrame(), validation, models)


@pytest.fixture
def comparison_references() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["A", "B"],
            "cutoff": [datetime(2026, 3, 23)] * 2,
            "season_length": [7, 7],
            "naive_mae": [10.0, 20.0],
            "naive_mse": [120.0, 450.0],
            "reference_status": ["ok", "ok"],
        }
    )


def test_comparison_references_accepts_different_order(
    comparison_references: pl.DataFrame,
) -> None:
    other = comparison_references.reverse()

    validate_comparison_references({"first": comparison_references, "second": other})


@pytest.mark.parametrize(
    ("column", "different_value"),
    [
        ("season_length", 3),
        ("naive_mae", 99.0),
        ("naive_mse", 999.0),
        ("reference_status", "zero_reference"),
    ],
)
def test_comparison_references_rejects_different_values(
    comparison_references: pl.DataFrame,
    column: str,
    different_value: int | float | str,
) -> None:
    other = comparison_references.with_columns(pl.lit(different_value).alias(column))

    with pytest.raises(ValueError, match="References differ.*second"):
        validate_comparison_references({"first": comparison_references, "second": other})


def test_comparison_references_rejects_missing_key(
    comparison_references: pl.DataFrame,
) -> None:
    other = comparison_references.head(1)

    with pytest.raises(ValueError, match="References differ.*second"):
        validate_comparison_references({"first": comparison_references, "second": other})


def test_comparison_references_rejects_duplicate_keys(
    comparison_references: pl.DataFrame,
) -> None:
    duplicate_row = comparison_references.head(1).with_columns(pl.lit(99.0).alias("naive_mae"))
    other = pl.concat([comparison_references, duplicate_row])

    with pytest.raises(ValueError, match="Duplicate references keys.*second"):
        validate_comparison_references({"first": comparison_references, "second": other})


def test_run_models_benchmark_rejects_different_references(
    monkeypatch: pytest.MonkeyPatch,
    comparison_evaluation: pl.DataFrame,
    comparison_references: pl.DataFrame,
) -> None:
    import forecasting.evaluation.benchmark as benchmark_module

    config: dict[str, object] = {
        "validation": {
            "train_start": "2026-03-16",
            "initial_train_days": 8,
            "first_cutoff": "2026-03-23",
            "last_cutoff": "2026-03-23",
            "horizon_days": 1,
            "step_days": 1,
            "test_start": "2026-03-25",
        }
    }
    validation = parse_validation_config(config)

    different_references = comparison_references.with_columns(pl.lit(99.0).alias("naive_mae"))

    def make_result(references: pl.DataFrame) -> benchmark_module.ModelBenchmarkResult:
        return benchmark_module.ModelBenchmarkResult(
            reports=pl.DataFrame(),
            forecasts=pl.DataFrame(),
            references=references,
            evaluation=comparison_evaluation,
            wape_scores=pl.DataFrame(),
            scaled_scores=pl.DataFrame(),
        )

    fake_results = iter(
        [
            make_result(comparison_references),
            make_result(different_references),
        ]
    )

    def fake_run_model_benchmark(
        data: pl.DataFrame,
        validation: benchmark_module.ValidationConfig,
        model_factory: Callable[[], ForecastModel],
    ) -> benchmark_module.ModelBenchmarkResult:
        return next(fake_results)

    monkeypatch.setattr(
        benchmark_module,
        "run_model_benchmark",
        fake_run_model_benchmark,
    )

    models: dict[str, Callable[[], ForecastModel]] = {
        "first": LastValueNaive,
        "second": LastValueNaive,
    }

    with pytest.raises(ValueError, match="References differ.*second"):
        run_models_benchmark(pl.DataFrame(), validation, models)
