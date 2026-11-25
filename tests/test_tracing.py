"""Tests for execution tracing and metadata capture."""

import hashlib
import json
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from forecasting.evaluation.tracing import (
    ExecutionMetadata,
    capture_metadata,
    get_git_info,
    get_installed_distributions,
    get_package_versions,
    hash_file_sha256,
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
    installed_distributions = [{"name": "example-package", "version": "1.2.3"}]
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.get_installed_distributions",
        lambda: installed_distributions,
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

    lockfile = tmp_path / "uv.lock"
    lockfile.write_bytes(b"test lockfile")

    metadata = capture_metadata(config_path, data_path, data, tmp_path)

    expected_lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    assert metadata.lockfile_sha256 == expected_lock_hash

    assert metadata.installed_distributions == installed_distributions
    assert metadata.git_dirty is git_dirty
    assert metadata.config_file == str(config_path)
    assert metadata.config_content == "test: value\n"
    assert metadata.data_file == str(data_path)
    assert metadata.data_sha256 == hashlib.sha256(data_path.read_bytes()).hexdigest()
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
        lockfile_sha256="b" * 64,
        installed_distributions=[{"name": "example-package", "version": "1.2.3"}],
        data_file="data.parquet",
        data_sha256="a" * 64,
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
    assert loaded["data_sha256"] == "a" * 64
    assert loaded["lockfile_sha256"] == "b" * 64
    assert loaded["installed_distributions"] == [{"name": "example-package", "version": "1.2.3"}]
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

    (tmp_path / "uv.lock").write_bytes(b"test lockfile")

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


def test_hash_file_sha256(tmp_path: Path) -> None:
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"abc")

    result = hash_file_sha256(data_path)

    assert result == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hash_file_sha256_changes_when_content_changes(tmp_path: Path) -> None:
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"abc")
    first_hash = hash_file_sha256(data_path)

    data_path.write_bytes(b"abd")
    second_hash = hash_file_sha256(data_path)

    assert first_hash != second_hash


def test_hash_file_sha256_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_file_sha256(tmp_path / "missing.bin")


def test_get_installed_distributions_returns_sorted_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_distributions = [
        SimpleNamespace(metadata={"Name": "PyYAML"}, version="6.0.3"),
        SimpleNamespace(metadata={"Name": "numpy"}, version="2.5.4"),
    ]
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.distributions",
        lambda: fake_distributions,
    )

    result = get_installed_distributions()

    assert result == [
        {"name": "numpy", "version": "2.5.4"},
        {"name": "PyYAML", "version": "6.0.3"},
    ]


def test_get_installed_distributions_rejects_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_distributions = [SimpleNamespace(metadata={}, version="1.0")]
    monkeypatch.setattr(
        "forecasting.evaluation.tracing.distributions",
        lambda: fake_distributions,
    )

    with pytest.raises(ValueError):
        get_installed_distributions()
