"""Tests for execution tracing and metadata capture."""

import json
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl
import pytest

from forecasting.evaluation.tracing import (
    ExecutionMetadata,
    capture_metadata,
    get_git_info,
    get_package_versions,
    is_git_dirty,
    save_metadata,
)


def test_get_git_info_returns_non_empty() -> None:
    """Git info should return something (not all unknowns)."""
    project_root = Path(__file__).resolve().parents[1]
    commit, branch = get_git_info(project_root)
    assert isinstance(commit, str)
    assert isinstance(branch, str)
    # At least one should not be "unknown" in a real repo
    assert commit != "unknown" or branch != "unknown"


def test_get_package_versions_includes_requested_packages() -> None:
    """Should return versions for requested packages."""
    versions = get_package_versions(["polars", "numpy"])

    assert "polars" in versions
    assert "numpy" in versions
    assert isinstance(versions["polars"], str)


@pytest.mark.parametrize("git_dirty", [True, False])
def test_capture_metadata_on_simple_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_dirty: bool,
) -> None:
    """Should capture metadata from config and data."""
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.get_git_info",
        lambda project_root: ("abc123", "test-branch"),
    )
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.is_git_dirty",
        lambda project_root: git_dirty,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: value\n")

    data_path = tmp_path / "data.parquet"
    data = pl.DataFrame(
        {
            "series_id": ["A", "A", "B"],
            "timestamp": [
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
                datetime(2026, 1, 3),
            ],
            "y": [1.0, 2.0, 3.0],
        }
    )
    data.write_parquet(data_path)

    metadata = capture_metadata(config_path, data_path, data, tmp_path)

    assert metadata.git_dirty is git_dirty
    assert metadata.config_file == str(config_path)
    assert metadata.config_content == "test: value\n"
    assert metadata.data_file == str(data_path)
    assert metadata.data_shape["n_rows"] == 3
    assert metadata.data_shape["n_series"] == 2
    assert metadata.data_shape["date_range_start"] == "2026-01-01T00:00:00"
    assert metadata.data_shape["date_range_end"] == "2026-01-03T00:00:00"


def test_save_metadata_to_json(tmp_path: Path) -> None:
    """Should save metadata as valid JSON."""
    metadata = ExecutionMetadata(
        timestamp="2026-11-23T10:00:00",
        git_commit="abc1234",
        git_branch="main",
        git_dirty=True,
        python_version="3.12.0",
        packages={"polars": "0.20.1"},
        config_file="config.yaml",
        config_content="test: value\n",
        data_file="data.parquet",
        data_shape={
            "n_rows": 100,
            "n_series": 10,
            "date_range_start": "2026-01-01",
            "date_range_end": "2026-12-31",
        },
    )

    output_path = tmp_path / "metadata.json"
    save_metadata(metadata, output_path)

    assert output_path.exists()

    # Verify it's valid JSON
    loaded = json.loads(output_path.read_text())
    assert loaded["git_commit"] == "abc1234"
    assert loaded["git_dirty"] is True
    assert loaded["data_shape"]["n_rows"] == 100


def test_capture_metadata_records_pyyaml_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.get_git_info",
        lambda project_root: ("abc123", "test-branch"),
    )
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.is_git_dirty",
        lambda project_root: False,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: value\n", encoding="utf-8")

    data_path = tmp_path / "data.parquet"
    data = pl.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [datetime(2026, 1, 1)],
            "y": [1.0],
        }
    )
    data.write_parquet(data_path)

    metadata = capture_metadata(config_path, data_path, data, tmp_path)

    assert metadata.packages["PyYAML"] == version("PyYAML")


def test_get_package_versions_marks_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr(
        "forecasting.evaluation.tracing.version",
        missing_version,
    )

    versions = get_package_versions(["missing-package"])

    assert versions == {"missing-package": "not installed"}


@pytest.mark.parametrize(
    ("git_output", "expected"),
    [
        ("", False),
        (" M scripts/run_benchmark.py\n", True),
        ("?? new_file.py\n", True),
    ],
)
def test_is_git_dirty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    git_output: str,
    expected: bool,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout=git_output,
            stderr="",
        )

    monkeypatch.setattr("forecasting.evaluation.tracing.subprocess.run", fake_run)

    assert is_git_dirty(tmp_path) is expected


def test_is_git_dirty_propagates_git_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def failing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["git", "status"])

    monkeypatch.setattr("forecasting.evaluation.tracing.subprocess.run", failing_run)

    with pytest.raises(subprocess.CalledProcessError):
        is_git_dirty(tmp_path)


@pytest.mark.parametrize("failing_call", [1, 2])
def test_get_git_info_propagates_git_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_call: int,
) -> None:
    call_count = 0

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1

        if call_count == failing_call:
            if kwargs.get("check") is True:
                raise subprocess.CalledProcessError(128, ["git"])
            return subprocess.CompletedProcess(
                args=["git"], returncode=128, stdout="", stderr="Git failed"
            )

        return subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="example\n", stderr=""
        )

    monkeypatch.setattr("forecasting.evaluation.tracing.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        get_git_info(tmp_path)
