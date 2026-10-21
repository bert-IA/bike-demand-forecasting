"""Download raw bike counter data from Paris open data."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

DATASET_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "comptage-velo-donnees-compteurs/exports/parquet"
)
RAW_FILENAME = "bike_counts.parquet"
METADATA_FILENAME = "bike_counts.meta.json"


def download_raw(
    dest_dir: Path,
    limit: int | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Download the dataset as Parquet into dest_dir and return the final file path.

    `limit` is for quick manual tests; None downloads everything.
    `client` can be injected (tests use a fake transport, so no network is needed).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / RAW_FILENAME
    tmp_path = final_path.with_suffix(".parquet.part")
    params: dict[str, int] = {} if limit is None else {"limit": limit}

    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True)
    try:
        # Stream to disk: the full file is never held in memory.
        with http.stream("GET", DATASET_URL, params=params) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        # Rename only after a complete download: no corrupted final file.
        tmp_path.replace(final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_client:
            http.close()

    try:
        _write_metadata(final_path, params)
    except BaseException:
        final_path.unlink(missing_ok=True)
        raise
    return final_path


def _write_metadata(final_path: Path, params: dict[str, int]) -> None:
    """Record where and when the data came from, for reproducibility."""
    row_count = int(pl.scan_parquet(final_path).select(pl.len()).collect().item())
    metadata = {
        "url": DATASET_URL,
        "params": params,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
    }
    meta_path = final_path.with_name(METADATA_FILENAME)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    path = download_raw(Path("data/raw"))
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
