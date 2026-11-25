"""Capture execution metadata for reproducibility and audit."""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl


@dataclass
class ExecutionMetadata:
    """Metadata about a benchmark execution."""

    timestamp: str  # ISO format
    git_commit: str  # Short hash or "unknown"
    git_branch: str  # Current branch
    git_dirty: bool
    python_version: str  # e.g., "3.12.0"
    packages: dict[str, str]  # e.g., {"polars": "0.20.1", ...}
    config_file: str  # Path to YAML
    config_content: str  # Full YAML content
    data_file: str  # Path to Parquet
    data_shape: dict[
        str, str | int
    ]  # {"n_rows": int, "n_series": int, "date_range_start": str, ...}


def get_git_info(project_root: Path) -> tuple[str, str]:
    """Return (commit_hash, branch_name)."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()[:8]

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()

    return commit, branch


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Return versions of specified distributions."""
    versions: dict[str, str] = {}

    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"

    return versions


def capture_metadata(
    config_path: Path,
    data_path: Path,
    data: pl.DataFrame,
    project_root: Path,
) -> ExecutionMetadata:
    """Capture comprehensive execution metadata."""
    config_content = config_path.read_text(encoding="utf-8")
    commit, branch = get_git_info(project_root)
    git_dirty = is_git_dirty(project_root)

    n_series = data.select("series_id").n_unique()
    date_range = data.select(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
    ).row(0)

    return ExecutionMetadata(
        timestamp=datetime.now().isoformat(),
        git_commit=commit,
        git_branch=branch,
        git_dirty=git_dirty,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        packages=get_package_versions(["polars", "statsmodels", "numpy", "PyYAML"]),
        config_file=str(config_path),
        config_content=config_content,
        data_file=str(data_path),
        data_shape={
            "n_rows": data.height,
            "n_series": n_series,
            "date_range_start": date_range[0].isoformat(),
            "date_range_end": date_range[1].isoformat(),
        },
    )


def save_metadata(metadata: ExecutionMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    output_path.write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )


def is_git_dirty(project_root: Path) -> bool:
    """Return whether Git detects changes in the project."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    return bool(result.stdout.strip())
