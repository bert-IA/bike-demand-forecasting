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

from datetime import datetime
from pathlib import Path

from forecasting.evaluation.benchmark import run_benchmark
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

    # 3. Run the benchmark
    print("⏳ Running benchmark...")
    start_time = datetime.now()

    config, results = run_benchmark(config_path, project_root, models)

    elapsed = datetime.now() - start_time
    print(f"✅ Benchmark completed in {elapsed.total_seconds():.1f}s")
    print()

    # 4. Display results
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Experiment: {config.experiment_name}")
    print(f"Data: {config.input_path}")
    print(f"Training start: {config.validation.train_start}")
    print(f"Initial training days: {config.validation.initial_train_days}")
    print(f"Validation period: {config.validation.first_cutoff} to {config.validation.last_cutoff}")
    print(f"Forecast horizon: {config.validation.horizon_days} days")
    print(f"Test period starts: {config.validation.test_start}")
    print()

    # 5. Summary by model
    print("-" * 70)
    print("SCORES BY MODEL")
    print("-" * 70)

    for model_name, result in results.items():
        wape_scores = result.wape_scores
        scaled_scores = result.scaled_scores

        mean_wape = wape_scores["wape"].mean()
        mean_mase = scaled_scores["mase"].mean()

        print(f"\n{model_name}:")
        print(f"  Mean WAPE: {mean_wape:.4f}")
        print(f"  Mean MASE: {mean_mase:.4f}")
        print(f"  Forecasts: {result.forecasts.height} observations")
        print(f"  Complete horizons: {(result.evaluation['y_true'].is_not_null().sum())}")

    print()
    print("-" * 70)
    print("RANKING BY MEAN MASE (lower is better)")
    print("-" * 70)

    rankings = []
    for model_name, result in results.items():
        mean_mase = result.scaled_scores["mase"].mean()
        rankings.append((model_name, mean_mase))

    rankings.sort(key=lambda x: x[1])
    for rank, (model_name, mean_mase) in enumerate(rankings, 1):
        print(f"{rank}. {model_name:30s} → MASE: {mean_mase:.4f}")

    print()

    # 6. Save results to Parquet
    output_dir = project_root / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, result in results.items():
        safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Save each DataFrame
        forecasts_path = output_dir / f"{timestamp}_{safe_name}_forecasts.parquet"
        wape_path = output_dir / f"{timestamp}_{safe_name}_wape.parquet"
        mase_path = output_dir / f"{timestamp}_{safe_name}_mase.parquet"

        result.forecasts.write_parquet(forecasts_path)
        result.wape_scores.write_parquet(wape_path)
        result.scaled_scores.write_parquet(mase_path)

        print(f"💾 Saved {model_name}:")
        print(f"   - {forecasts_path.name}")
        print(f"   - {wape_path.name}")
        print(f"   - {mase_path.name}")

    print()
    print("=" * 70)
    print("✅ Benchmark execution and results saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
