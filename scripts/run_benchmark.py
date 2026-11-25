#!/usr/bin/env python3
"""
Execute the bike demand forecasting benchmark with baseline models.

This script:
1. Loads the benchmark configuration from YAML
2. Loads the processed bike daily data
3. Runs the benchmark with three baseline models
4. Displays results and saves them to Parquet files

Usage:
    uv run python scripts/run_benchmark.py
"""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from forecasting.evaluation.benchmark import (
    load_benchmark_config,
    load_benchmark_data,
    parse_benchmark_config,
    run_benchmark,
)
from forecasting.evaluation.tracing import capture_metadata, save_metadata
from forecasting.models.exponential_smoothing import SeasonalExponentialSmoothing
from forecasting.models.last_value import LastValueNaive
from forecasting.models.seasonal_naive import SeasonalNaive


def main() -> None:
    """Execute the benchmark and save results."""

    # 1. Locate project root and config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "benchmark_bike_validation.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"📋 Configuration: {config_path}")
    print(f"📊 Project root: {project_root}")
    print()

    # 2. Define baseline models
    models = {
        "last_value": LastValueNaive,
        "seasonal_naive_7": lambda: SeasonalNaive(season_length=7),
        "exponential_smoothing_7": lambda: SeasonalExponentialSmoothing(season_length=7),
    }

    print("🤖 Models to benchmark:")
    for name in models:
        print(f"   - {name}")
    print()

    config_load = load_benchmark_config(config_path)
    benchmark_config = parse_benchmark_config(config_load)
    data = load_benchmark_data(project_root, benchmark_config.input_path)

    metadata = capture_metadata(
        config_path, project_root / benchmark_config.input_path, data, project_root
    )

    # 3. Run the benchmark
    print("⏳ Running benchmark...")
    start_time = datetime.now()

    config_returned, results = run_benchmark(benchmark_config, data, models)

    elapsed = datetime.now() - start_time
    print(f"✅ Benchmark completed in {elapsed.total_seconds():.1f}s")
    print()

    # 4. Display results
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Experiment: {config_returned.experiment_name}")
    print(f"Data: {config_returned.input_path}")
    print(f"Training start: {config_returned.validation.train_start}")
    print(f"Initial training days: {config_returned.validation.initial_train_days}")
    print(
        f"Validation cutoffs: {config_returned.validation.first_cutoff} to {
            config_returned.validation.last_cutoff
        }"
    )
    print(f"Forecast horizon: {config_returned.validation.horizon_days} days")
    print(f"Test period starts: {config_returned.validation.test_start}")
    print()

    # 5. Summary by model
    print("-" * 70)
    print("SCORES BY MODEL")
    print("-" * 70)

    horizon = config_returned.validation.horizon_days
    for model_name, result in results.items():
        wape_scores = result.wape_scores
        scaled_scores = result.scaled_scores

        complete_horizons = wape_scores.filter(
            (pl.col("n_forecasts") == horizon) & (pl.col("n_evaluable") == horizon)
        ).height

        valid_wape = wape_scores["wape"].is_not_null().sum()
        valid_mase = scaled_scores["mase"].is_not_null().sum()
        valid_rmsse = scaled_scores["rmsse"].is_not_null().sum()

        mean_wape = wape_scores["wape"].mean()
        mean_mase = scaled_scores["mase"].mean()

        print(f"\n{model_name}:")
        print(f"  Valid WAPE scores: {valid_wape}/{wape_scores.height}")
        print(f"  Valid MASE scores: {valid_mase}/{scaled_scores.height}")
        print(f"  Valid RMSSE scores: {valid_rmsse}/{scaled_scores.height}")

        if mean_wape is None:
            print("  Mean WAPE: N/A (no valid scores)")
        else:
            print(f"  Mean WAPE: {mean_wape:.4f}")
        if mean_mase is None:
            print("  Mean MASE: N/A (no valid scores)")
        else:
            print(f"  Mean MASE: {mean_mase:.4f}")

        print(f"  Forecasts: {result.forecasts.height} observations")
        print(f"  Evaluable forecast rows : {(result.evaluation['y_true'].is_not_null().sum())}")
        print(f"  complete horizons : {complete_horizons}")

    print()
    print("-" * 70)
    print("RANKING BY MEAN MASE (lower is better)")
    print("-" * 70)

    rankings: list[tuple[str, float]] = []
    unavailable_models: list[str] = []
    for model_name, result in results.items():
        mean_mase = result.scaled_scores["mase"].mean()

        if mean_mase is None:
            unavailable_models.append(model_name)
        else:
            rankings.append((model_name, mean_mase))

    if unavailable_models:
        print(f"unavailable models for ranking : {unavailable_models}")

    else:
        rankings.sort(key=lambda x: x[1])
        for rank, (model_name, mean_mase) in enumerate(rankings, 1):
            print(f"{rank}. {model_name:30s} → MASE: {mean_mase:.4f}")

    print()

    # 6. Save results to Parquet
    output_dir = project_root / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(tempfile.mkdtemp(dir=output_dir, prefix=f"run-{timestamp}-"))

    print(f"Results directory: {output_dir}")

    for model_name, result in results.items():
        safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Save each DataFrame
        forecasts_path = output_dir / f"{timestamp}_{safe_name}_forecasts.parquet"
        wape_path = output_dir / f"{timestamp}_{safe_name}_wape.parquet"
        mase_path = output_dir / f"{timestamp}_{safe_name}_mase.parquet"
        reports_path = output_dir / f"{timestamp}_{safe_name}_reports.parquet"
        references_path = output_dir / f"{timestamp}_{safe_name}_references.parquet"
        evaluation_path = output_dir / f"{timestamp}_{safe_name}_evaluation.parquet"

        result.forecasts.write_parquet(forecasts_path)
        result.reports.write_parquet(reports_path)
        result.references.write_parquet(references_path)
        result.evaluation.write_parquet(evaluation_path)
        result.wape_scores.write_parquet(wape_path)
        result.scaled_scores.write_parquet(mase_path)

        print(f"💾 Saved {model_name}:")
        print(f"   - {forecasts_path.name}")
        print(f"   - {reports_path.name}")
        print(f"   - {references_path.name}")
        print(f"   - {evaluation_path.name}")
        print(f"   - {wape_path.name}")
        print(f"   - {mase_path.name}")

    output_path = output_dir / f"{timestamp}_metadata.json"
    save_metadata(metadata, output_path)  # ← pas d'assignation
    print("💾 Saved execution metadata:")
    print(f"   - {output_path.name}")

    print()
    print("=" * 70)
    print("✅ Benchmark execution and results saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
