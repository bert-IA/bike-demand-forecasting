"""Capture execution metadata for reproducibility and audit."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distributions, version
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
    lockfile_sha256: str
    installed_distributions: list[dict[str, str]]
    data_sha256: str
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


def get_installed_distributions() -> list[dict[str, str]]:
    """Return visible Python distributions and their versions."""
    results = []

    for distribution in distributions():
        name = distribution.metadata.get("Name")
        package_version = distribution.version
        if not name or not package_version:
            raise ValueError(f"Missing name or version: name={name!r}, version={package_version!r}")
        results.append({"name": name, "version": package_version})
    results.sort(key=lambda item: (item["name"].casefold(), item["version"]))
    return results


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
    lockfile = project_root / "uv.lock"
    lockfile_sha256 = hash_file_sha256(lockfile)
    n_series = data.select("series_id").n_unique()
    data_sha256 = hash_file_sha256(data_path)
    date_range = data.select(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
    ).row(0)
    installed_distributions = get_installed_distributions()

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
        lockfile_sha256=lockfile_sha256,
        data_sha256=data_sha256,
        data_shape={
            "n_rows": data.height,
            "n_series": n_series,
            "date_range_start": date_range[0].isoformat(),
            "date_range_end": date_range[1].isoformat(),
        },
        installed_distributions=installed_distributions,
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


def hash_file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()
