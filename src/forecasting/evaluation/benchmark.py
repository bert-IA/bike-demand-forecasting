from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import yaml

from forecasting.data.schema import validate_series
from forecasting.evaluation.backtest import (
    attach_actuals_to_forecasts,
    evaluate_scaled_metrics_by_series,
    evaluate_wape_by_series,
    run_backtest,
)
from forecasting.models.protocol import ForecastModel


def load_benchmark_config(path: Path) -> dict[str, object]:
    """Load a benchmark configuration from a YAML file."""
    content = path.read_text(encoding="utf-8")
    data: object = yaml.safe_load(content)

    if not isinstance(data, dict):
        raise ValueError("benchmark configuration must be a mapping")

    if not all(isinstance(key, str) for key in data):
        raise ValueError("benchmark configuration keys must be strings")

    return data


def get_validation_section(config: dict[str, object]) -> dict[str, object]:
    """Return the validation section of the benchmark configuration."""
    if "validation" not in config:
        raise ValueError("benchmark configuration is missing validation")

    section = config["validation"]

    if not isinstance(section, dict):
        raise ValueError("validation must be a mapping")

    if not all(isinstance(key, str) for key in section):
        raise ValueError("validation keys must be string")

    return section


def get_positive_int(section: dict[str, object], name: str) -> int:
    """Return a strictly positive integer from a configuration section."""

    if name not in section:
        raise ValueError(f"{name} is missing in benchmark configuration")

    value = section[name]

    if (not isinstance(value, int)) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive integer")

    return value


def get_config_date(section: dict[str, object], name: str) -> date:
    """Return a date from a configuration section."""

    if name not in section:
        raise ValueError(f"{name} is missing in benchmark configuration")

    value = section[name]

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string in YYYY-MM-DD format")

    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid date in YYYY-MM-DD format") from error

    if parsed_date.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD format")

    return parsed_date


def validate_validation_window(
    first_cutoff: date,
    last_cutoff: date,
    horizon_days: int,
    test_start: date,
) -> None:
    """Check that the validation window is ordered and avoids the final test."""

    if first_cutoff > last_cutoff:
        raise ValueError("first_cutoff must be before or equal to last_cutoff")

    last_forecast_day = last_cutoff + timedelta(days=horizon_days)

    if last_forecast_day >= test_start:
        raise ValueError(
            f"last forecast day ({last_forecast_day}) must be before test_start ({test_start})"
        )


def build_cutoffs(
    first_cutoff: date,
    last_cutoff: date,
    step_days: int,
) -> list[date]:
    """Build regularly spaced cutoffs, including both endpoints."""

    if not isinstance(step_days, int) or isinstance(step_days, bool) or step_days <= 0:
        raise ValueError("step_days must be a positive integer")

    if first_cutoff > last_cutoff:
        raise ValueError("first_cutoff must be before or equal to last_cutoff")

    duration_days = (last_cutoff - first_cutoff).days
    if duration_days % step_days != 0:
        raise ValueError("last_cutoff must align with first_cutoff and step_days")

    cutoffs = []
    current_cutoff = first_cutoff
    while current_cutoff <= last_cutoff:
        cutoffs.append(current_cutoff)

        if current_cutoff == last_cutoff:
            break

        current_cutoff = current_cutoff + timedelta(days=step_days)

    return cutoffs


def get_validation_cutoffs(config: dict[str, object]) -> list[date]:
    validation_config = parse_validation_config(config)

    return list(validation_config.cutoffs)


@dataclass(frozen=True)
class ValidationConfig:
    train_start: date
    initial_train_days: int
    first_cutoff: date
    last_cutoff: date
    horizon_days: int
    step_days: int
    test_start: date
    cutoffs: tuple[date, ...]


def parse_validation_config(config: dict[str, object]) -> ValidationConfig:
    """Parse and validate the benchmark validation settings."""

    section = get_validation_section(config)

    train_start = get_config_date(section, "train_start")
    initial_train_days = get_positive_int(section, "initial_train_days")

    first_cutoff = get_config_date(section, "first_cutoff")
    if train_start > first_cutoff:
        raise ValueError("train_start must be before or equal to first_cutoff")
    last_cutoff = get_config_date(section, "last_cutoff")
    test_start = get_config_date(section, "test_start")

    if (first_cutoff - train_start).days + 1 < initial_train_days:
        raise ValueError("training window is shorter than initial_train_days")

    horizon_days = get_positive_int(section, "horizon_days")
    step_days = get_positive_int(section, "step_days")

    validate_validation_window(
        first_cutoff,
        last_cutoff,
        horizon_days,
        test_start,
    )

    cutoffs = build_cutoffs(first_cutoff, last_cutoff, step_days)

    return ValidationConfig(
        train_start,
        initial_train_days,
        first_cutoff=first_cutoff,
        last_cutoff=last_cutoff,
        horizon_days=horizon_days,
        step_days=step_days,
        test_start=test_start,
        cutoffs=tuple(cutoffs),
    )


def cutoffs_to_datetimes(cutoffs: tuple[date, ...]) -> list[datetime]:
    """Convert cutoff dates to naive datetimes at midnight."""

    result: list[datetime] = []
    result.extend(datetime.combine(cutoff, time.min) for cutoff in cutoffs)
    return result


def get_section_string(
    config: dict[str, object],
    section_name: str,
    key_name: str,
) -> str:
    """Return a non-empty string from a configuration section."""

    if section_name not in config:
        raise ValueError(f"{section_name} is missing from configuration")

    section = config[section_name]

    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be a mapping")

    if key_name not in section:
        raise ValueError(f"{section_name}.{key_name} is missing")

    value = section[key_name]

    if not isinstance(value, str):
        raise ValueError(f"{section_name}.{key_name} must be a string")

    if not value.strip():
        raise ValueError(f"{section_name}.{key_name} must not be empty")

    return value


def get_experiment_name(config: dict[str, object]) -> str:
    """Return the experiment name from the benchmark configuration."""
    return get_section_string(config, "experiment", "name")


def get_data_input_path(config: dict[str, object]) -> str:
    """Return the input data path from the benchmark configuration."""
    return get_section_string(config, "data", "input_path")


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment_name: str
    input_path: str
    validation: ValidationConfig


def parse_benchmark_config(config: dict[str, object]) -> BenchmarkConfig:
    """Parse and validate the benchmark configuration."""

    experience = get_experiment_name(config)
    input_path = get_data_input_path(config)
    validation_config = parse_validation_config(config)

    return BenchmarkConfig(
        experiment_name=experience,
        input_path=input_path,
        validation=validation_config,
    )


def load_benchmark_data(project_root: Path, input_path: str) -> pl.DataFrame:
    """Load benchmark data from a Parquet file."""

    if not project_root.is_absolute():
        raise ValueError(f"project_root is not an absolute path, got {project_root}")

    data_path = project_root / input_path

    if not data_path.is_file():
        raise FileNotFoundError(f"{data_path} is not a file")

    return validate_series(pl.read_parquet(data_path))


def get_validation_data(data: pl.DataFrame, train_start: date, test_start: date) -> pl.DataFrame:
    """Return observations from train_start up to, but excluding, test_start."""

    train_start_datetime = datetime.combine(train_start, time.min)
    test_start_datetime = datetime.combine(test_start, time.min)

    return data.filter(
        (pl.col("timestamp") >= train_start_datetime) & (pl.col("timestamp") < test_start_datetime)
    )


def run_validation_backtest(
    data: pl.DataFrame,
    validation: ValidationConfig,
    model_factory: Callable[[], ForecastModel],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run one model on the configured validation period."""

    filtered_data = get_validation_data(
        data,
        validation.train_start,
        validation.test_start,
    )

    cutoffs = cutoffs_to_datetimes(validation.cutoffs)

    return run_backtest(
        filtered_data,
        cutoffs,
        model_factory,
        horizon=validation.horizon_days,
        initial_train_days=validation.initial_train_days,
    )


def evaluate_validation_forecasts(
    data: pl.DataFrame,
    validation: ValidationConfig,
    forecasts: pl.DataFrame,
    references: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Evaluate forecasts against observations from the validation period."""

    validation_data = get_validation_data(data, validation.train_start, validation.test_start)

    evaluation = attach_actuals_to_forecasts(forecasts, validation_data)

    wape_scores = evaluate_wape_by_series(evaluation)
    scaled_scores = evaluate_scaled_metrics_by_series(evaluation, references)

    return evaluation, wape_scores, scaled_scores


@dataclass(frozen=True)
class ModelBenchmarkResult:
    reports: pl.DataFrame
    forecasts: pl.DataFrame
    references: pl.DataFrame
    evaluation: pl.DataFrame
    wape_scores: pl.DataFrame
    scaled_scores: pl.DataFrame


def run_model_benchmark(
    data: pl.DataFrame,
    validation: ValidationConfig,
    model_factory: Callable[[], ForecastModel],
) -> ModelBenchmarkResult:
    """Run and evaluate one model on the validation period."""

    reports, forecasts, references = run_validation_backtest(data, validation, model_factory)

    evaluation, wape_scores, scaled_scores = evaluate_validation_forecasts(
        data, validation, forecasts, references
    )

    return ModelBenchmarkResult(
        reports, forecasts, references, evaluation, wape_scores, scaled_scores
    )


def run_models_benchmark(
    data: pl.DataFrame,
    validation: ValidationConfig,
    models: dict[str, Callable[[], ForecastModel]],
) -> dict[str, ModelBenchmarkResult]:
    """Run and evaluate each model in the benchmark."""

    results: dict[str, ModelBenchmarkResult] = {}

    for name, factory in models.items():
        results[name] = run_model_benchmark(data, validation, factory)

    evaluations: dict[str, pl.DataFrame] = {}
    for name, result in results.items():
        evaluations[name] = result.evaluation

    validate_comparison_perimeter(evaluations)

    references: dict[str, pl.DataFrame] = {}
    for name, result in results.items():
        references[name] = result.references

    validate_comparison_references(references)

    return results


def run_benchmark(
    benchmark_config: BenchmarkConfig,
    data: pl.DataFrame,
    models: dict[str, Callable[[], ForecastModel]],
) -> tuple[BenchmarkConfig, dict[str, ModelBenchmarkResult]]:
    benchmark = run_models_benchmark(data, benchmark_config.validation, models)
    return benchmark_config, benchmark


def validate_comparison_perimeter(evaluations: dict[str, pl.DataFrame]) -> None:
    """Require identical evaluation rows across models."""
    keys = ["series_id", "cutoff", "timestamp"]
    reference: pl.DataFrame | None = None

    for model_name, evaluation in evaluations.items():
        selected = evaluation.select(
            *keys,
            "y_true",
            "is_evaluable",
        )

        if selected.select(keys).is_duplicated().any():
            raise ValueError(f"Duplicate forecast keys for model {model_name!r}")

        selected = selected.sort(keys)

        if reference is None:
            reference = selected
        elif not reference.equals(selected):
            raise ValueError(f"Evaluation perimeter differs for model {model_name!r}")


def validate_comparison_references(
    references: dict[str, pl.DataFrame],
) -> None:
    """Require identical seasonal references across models."""

    keys = ["series_id", "cutoff"]
    reference_table: pl.DataFrame | None = None

    for model_name, model_references in references.items():
        selected = model_references.select(
            *keys,
            "season_length",
            "naive_mae",
            "naive_mse",
            "reference_status",
        )

        selected = selected.sort(keys)

        if selected.select(keys).is_duplicated().any():
            raise ValueError(f"Duplicate references keys for model {model_name!r}")

        if reference_table is None:
            reference_table = selected
        elif not reference_table.equals(selected):
            raise ValueError(f"References differ for model {model_name!r}")


@dataclass(frozen=True)
class SARIMAConfig:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str
    normalize: bool
    optimizer: str
    maxiter: int


def parse_sarima_config(config: dict[str, object]) -> SARIMAConfig:
    """Parse and validate the SARIMA model settings."""
    ...
    if "models" not in config:
        raise ValueError("section model is missing from configuration")

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping")

    sarima = models.get("sarima")
    if not isinstance(sarima, dict):
        raise ValueError("models.sarima must be a mapping")

    if not all(isinstance(key, str) for key in sarima):
        raise ValueError("models.sarima keys must be strings")

    def read_order(name: str, length: int) -> list[int]:
        value = sarima.get(name)

        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"{name} must be a list of {length} non-negative integers")

        result: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} must contain non-negative integers")
            result.append(item)

        return result

    order = read_order("order", 3)
    seasonal = read_order("seasonal_order", 4)

    period = seasonal[3]
    if period == 1 or (period == 0 and any(seasonal[:3])):
        raise ValueError("seasonal_order period must be at least 2 when seasonal terms are used")

    trend = sarima.get("trend")
    if not isinstance(trend, str) or trend not in ("n", "c", "t", "ct"):
        raise ValueError("trend must be one of n, c, t, ct")

    normalize = sarima.get("normalize")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean")

    optimizer = sarima.get("optimizer")
    optimizers = (
        "newton",
        "nm",
        "bfgs",
        "lbfgs",
        "powell",
        "cg",
        "ncg",
        "basinhopping",
        "minimize",
    )
    if not isinstance(optimizer, str) or optimizer not in optimizers:
        raise ValueError("optimizer must be a supported statsmodels method")

    maxiter = get_positive_int(sarima, "maxiter")

    return SARIMAConfig(
        order=(order[0], order[1], order[2]),
        seasonal_order=(seasonal[0], seasonal[1], seasonal[2], seasonal[3]),
        trend=trend,
        normalize=normalize,
        optimizer=optimizer,
        maxiter=maxiter,
    )
