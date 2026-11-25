import json
import runpy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from forecasting.evaluation.benchmark import SARIMAConfig
from forecasting.models.protocol import ForecastModel
from forecasting.models.sarima import SeasonalARIMA


def test_mean_score_handles_missing_and_invalid_values() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    mean_score = runpy.run_path(str(script_path))["_mean_score"]

    assert mean_score(pl.Series("wape", [0.1, None, 0.3])) == pytest.approx(0.2)
    assert mean_score(pl.Series("mase", [None], dtype=pl.Float64)) is None
    assert mean_score(pl.Series("mase", [None])) is None
    with pytest.raises(TypeError, match="must be numeric"):
        mean_score(pl.Series("wape", ["invalid"]))


def test_runs_save_results_in_separate_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    main = runpy.run_path(str(script_path))["main"]
    script_globals = main.__globals__

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "benchmark_bike_validation.yaml").write_text("test: true\n")
    monkeypatch.setitem(script_globals, "__file__", str(tmp_path / "scripts" / "run_benchmark.py"))

    class FixedClock:
        @staticmethod
        def now() -> datetime:
            return datetime(2026, 1, 1, 12, 0, 0)

    @dataclass
    class FakeMetadata:
        model_configs: dict[str, dict[str, object]]
        marker: str = "test"

    config = SimpleNamespace(
        experiment_name="test",
        input_path="data.parquet",
        validation=SimpleNamespace(
            train_start="2026-01-01",
            initial_train_days=14,
            first_cutoff="2026-01-14",
            last_cutoff="2026-01-14",
            horizon_days=1,
            test_start="2026-01-16",
        ),
    )
    result = SimpleNamespace(
        reports=pl.DataFrame({"series_id": ["A"], "eligible": [True]}),
        references=pl.DataFrame({"series_id": ["A"], "naive_mae": [2.0]}),
        forecasts=pl.DataFrame({"y_pred": [10.0, 10.0, 10.0, 10.0]}),
        evaluation=pl.DataFrame({"y_true": [11.0, None, 12.0, 13.0]}),
        wape_scores=pl.DataFrame(
            {
                "wape": [0.1, None, 0.2],
                "n_forecasts": [1, 1, 2],
                "n_evaluable": [1, 0, 2],
            }
        ),
        scaled_scores=pl.DataFrame({"mase": [0.2, None, 0.3], "rmsse": [0.4, None, None]}),
    )
    sarima_settings: dict[str, object] = {
        "order": [0, 0, 1],
        "seasonal_order": [0, 0, 1, 7],
        "trend": "n",
        "normalize": False,
        "optimizer": "lbfgs",
        "maxiter": 123,
    }

    def fake_capture_metadata(
        *args: object,
        model_configs: dict[str, dict[str, object]],
    ) -> FakeMetadata:
        return FakeMetadata(model_configs=model_configs)

    def fake_run_benchmark(
        benchmark_config: object,
        data: object,
        models: dict[str, Callable[[], ForecastModel]],
    ) -> tuple[object, dict[str, SimpleNamespace]]:
        model = models["sarima"]()
        assert isinstance(model, SeasonalARIMA)
        assert model.order == (0, 0, 1)
        assert model.seasonal_order == (0, 0, 1, 7)
        assert model.trend == "n"
        assert model.normalize is False
        assert model.optimizer == "lbfgs"
        assert model.maxiter == 123

        return benchmark_config, {name: result for name in models}

    monkeypatch.setitem(
        script_globals,
        "load_benchmark_config",
        lambda path: {"models": {"sarima": sarima_settings}},
    )
    monkeypatch.setitem(script_globals, "datetime", FixedClock)
    monkeypatch.setitem(script_globals, "parse_benchmark_config", lambda raw: config)
    monkeypatch.setitem(script_globals, "load_benchmark_data", lambda root, path: None)
    monkeypatch.setitem(script_globals, "capture_metadata", fake_capture_metadata)
    monkeypatch.setitem(script_globals, "run_benchmark", fake_run_benchmark)

    main()
    results_dir = tmp_path / "data" / "results"
    first_dir = next(results_dir.iterdir())
    first_files = {file.name: file.read_bytes() for file in first_dir.iterdir()}

    main()
    output = capsys.readouterr().out
    assert output.count("Valid WAPE scores: 2/3") == 8
    assert output.count("Valid MASE scores: 2/3") == 8
    assert output.count("Valid RMSSE scores: 1/3") == 8
    assert output.count("complete horizons : 1") == 8
    directories = list(results_dir.iterdir())
    assert len(directories) == 2
    assert all(directory.is_dir() for directory in directories)
    assert {file.name: file.read_bytes() for file in first_dir.iterdir()} == first_files

    model_names = [
        "last_value",
        "seasonal_naive_7",
        "exponential_smoothing_7",
        "sarima",
    ]
    expected_files = {"20260101_120000_metadata.json"}
    expected_files.update(
        f"20260101_120000_{model}_{kind}.parquet"
        for model in model_names
        for kind in ("forecasts", "wape", "mase", "reports", "references", "evaluation")
    )
    for directory in directories:
        assert {file.name for file in directory.iterdir()} == expected_files
        metadata = json.loads((directory / "20260101_120000_metadata.json").read_text())
        assert metadata["marker"] == "test"
        assert metadata["model_configs"]["sarima"] == sarima_settings
        for model in model_names:
            for kind in ("reports", "references", "evaluation"):
                path = directory / f"20260101_120000_{model}_{kind}.parquet"
                assert pl.read_parquet(path).equals(getattr(result, kind))


def test_missing_scores_skip_ranking_and_save_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    main = runpy.run_path(str(script_path))["main"]
    script_globals = main.__globals__

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "benchmark_bike_validation.yaml").write_text("test: true\n")
    monkeypatch.setitem(script_globals, "__file__", str(tmp_path / "scripts" / "run_benchmark.py"))

    @dataclass
    class FakeMetadata:
        marker: str = "test"

    config = SimpleNamespace(
        experiment_name="test",
        input_path="data.parquet",
        validation=SimpleNamespace(
            train_start="2026-01-01",
            initial_train_days=14,
            first_cutoff="2026-01-14",
            last_cutoff="2026-01-14",
            horizon_days=1,
            test_start="2026-01-16",
        ),
    )

    def make_result(wape: float | None, mase: float | None) -> SimpleNamespace:
        return SimpleNamespace(
            reports=pl.DataFrame({"series_id": ["A"], "eligible": [True]}),
            references=pl.DataFrame({"series_id": ["A"], "naive_mae": [2.0]}),
            forecasts=pl.DataFrame({"y_pred": [10.0]}),
            evaluation=pl.DataFrame({"y_true": [11.0]}),
            wape_scores=pl.DataFrame(
                {
                    "wape": pl.Series([wape], dtype=pl.Float64),
                    "n_forecasts": [1],
                    "n_evaluable": [1],
                }
            ),
            scaled_scores=pl.DataFrame(
                {
                    "mase": pl.Series([mase], dtype=pl.Float64),
                    "rmsse": pl.Series([None], dtype=pl.Float64),
                }
            ),
        )

    results = {
        "last_value": make_result(None, None),
        "seasonal_naive_7": make_result(0.1, 0.2),
        "exponential_smoothing_7": make_result(0.2, 0.3),
        "sarima": make_result(0.15, 0.25),
    }

    monkeypatch.setitem(script_globals, "load_benchmark_config", lambda path: {})
    monkeypatch.setitem(script_globals, "parse_benchmark_config", lambda raw: config)
    monkeypatch.setitem(
        script_globals,
        "parse_sarima_config",
        lambda raw: SARIMAConfig(
            order=(1, 0, 0),
            seasonal_order=(1, 1, 1, 7),
            trend="c",
            normalize=True,
            optimizer="powell",
            maxiter=200,
        ),
    )
    monkeypatch.setitem(script_globals, "load_benchmark_data", lambda root, path: None)
    monkeypatch.setitem(script_globals, "capture_metadata", lambda *args, **kwargs: FakeMetadata())
    monkeypatch.setitem(
        script_globals,
        "run_benchmark",
        lambda config, data, models: (config, results),
    )

    main()

    output = capsys.readouterr().out
    assert "Mean WAPE: N/A (no valid scores)" in output
    assert "Mean MASE: N/A (no valid scores)" in output
    assert "unavailable models for ranking" in output
    assert "last_value" in output
    assert "\n1. " not in output

    directories = list((tmp_path / "data" / "results").iterdir())
    assert len(directories) == 1
    assert len(list(directories[0].iterdir())) == 25
