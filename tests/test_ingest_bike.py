"""Tests for the raw bike data download (no network access)."""

import io
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import polars as pl
import pytest

from forecasting.data.ingest_bike import (
    METADATA_FILENAME,
    RAW_FILENAME,
    download_raw,
)


def make_parquet_bytes(rows: int = 3) -> bytes:
    """Build a tiny valid Parquet file in memory."""
    buffer = io.BytesIO()
    pl.DataFrame({"id_compteur": ["a"] * rows, "sum_counts": [1.0] * rows}).write_parquet(buffer)
    return buffer.getvalue()


def make_client(*, status: int = 200, content: bytes = b"") -> httpx.Client:
    """Return a client wired to a fake server that always answers the same thing."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(respond))


def make_recording_client() -> tuple[httpx.Client, list[httpx.Request]]:
    """Return a client and the list where the fake server records every request it gets."""
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=make_parquet_bytes())

    return httpx.Client(transport=httpx.MockTransport(respond)), seen


def test_download_writes_file_and_metadata(tmp_path: Path) -> None:
    client = make_client(content=make_parquet_bytes(rows=3))

    result = download_raw(tmp_path, client=client)

    assert result == tmp_path / RAW_FILENAME
    assert result.exists()
    assert not result.with_suffix(".parquet.part").exists()
    meta = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["row_count"] == 3


def test_limit_is_sent_as_query_parameter(tmp_path: Path) -> None:
    client, seen = make_recording_client()

    download_raw(tmp_path, limit=10, client=client)

    assert len(seen) == 1
    assert seen[0].url.params["limit"] == "10"


def test_no_limit_sends_no_query_parameter(tmp_path: Path) -> None:
    client, seen = make_recording_client()

    download_raw(tmp_path, client=client)

    assert len(seen) == 1
    assert "limit" not in seen[0].url.params


def test_http_error_leaves_no_file(tmp_path: Path) -> None:
    client = make_client(status=500)

    with pytest.raises(httpx.HTTPStatusError):
        download_raw(tmp_path, client=client)

    final_path = tmp_path / RAW_FILENAME
    assert not final_path.exists()
    assert not final_path.with_suffix(".parquet.part").exists()


def test_invalid_content_leaves_no_file(tmp_path: Path) -> None:
    # A 200 response whose body is not Parquet: the row count read must fail.
    client = make_client(content=b"<html>not parquet</html>")

    with pytest.raises(pl.exceptions.PolarsError):
        download_raw(tmp_path, client=client)

    assert not (tmp_path / RAW_FILENAME).exists()


def test_failure_during_download_removes_part_file(tmp_path: Path) -> None:
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b"partial data"
            raise httpx.ReadError("connection lost")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream())

    client = httpx.Client(transport=httpx.MockTransport(respond))

    with pytest.raises(httpx.ReadError):
        download_raw(tmp_path, client=client)

    assert not (tmp_path / RAW_FILENAME).exists()
    assert not (tmp_path / f"{RAW_FILENAME}.part").exists()
