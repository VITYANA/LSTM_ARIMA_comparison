"""Load reproducible raw-data snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

REQUIRED_PRICE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
)


@dataclass(frozen=True)
class SnapshotMetadata:
    """Research metadata read from the snapshot manifest."""

    created_at_utc: str
    tickers: tuple[str, ...]
    start: str
    end: str
    interval: str
    source: str


@dataclass(frozen=True)
class RawSnapshot:
    """Verified prices together with their research metadata."""

    prices: pd.DataFrame
    metadata: SnapshotMetadata


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    """Return a manifest section as a mapping or fail clearly."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest section {name!r} must be an object")
    return cast(Mapping[str, object], value)


def _as_string(value: object, name: str) -> str:
    """Return a manifest field as a string or fail clearly."""
    if not isinstance(value, str):
        raise ValueError(f"Manifest field {name!r} must be a string")
    return value


def _sha256(path: Path) -> str:
    """Calculate a file digest independently of the download module."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_snapshot(raw_dir: Path = Path("data/raw")) -> RawSnapshot:
    """Load the combined CSV after verifying it against the manifest."""
    manifest_path = raw_dir / "manifest.json"
    manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _as_mapping(manifest_value, "root")
    config = _as_mapping(manifest.get("config"), "config")
    files = _as_mapping(manifest.get("files"), "files")
    prices_record = _as_mapping(files.get("prices.csv"), "files.prices.csv")

    prices_path = raw_dir / "prices.csv"
    expected_digest = _as_string(prices_record.get("sha256"), "files.prices.csv.sha256")
    if _sha256(prices_path) != expected_digest:
        raise ValueError("prices.csv SHA-256 does not match manifest")

    expected_bytes = prices_record.get("bytes")
    if not isinstance(expected_bytes, int) or prices_path.stat().st_size != expected_bytes:
        raise ValueError("prices.csv byte count does not match manifest")

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    missing_columns = set(REQUIRED_PRICE_COLUMNS) - set(prices.columns)
    if missing_columns:
        raise ValueError(f"prices.csv missing columns: {sorted(missing_columns)}")

    expected_rows = prices_record.get("rows")
    if not isinstance(expected_rows, int) or len(prices) != expected_rows:
        raise ValueError("prices.csv row count does not match manifest")

    ticker_values = config.get("tickers")
    if not isinstance(ticker_values, list) or not all(
        isinstance(ticker, str) for ticker in ticker_values
    ):
        raise ValueError("Manifest field 'config.tickers' must be a string list")

    metadata = SnapshotMetadata(
        created_at_utc=_as_string(manifest.get("created_at_utc"), "created_at_utc"),
        tickers=tuple(cast(list[str], ticker_values)),
        start=_as_string(config.get("start"), "config.start"),
        end=_as_string(config.get("end"), "config.end"),
        interval=_as_string(config.get("interval"), "config.interval"),
        source=_as_string(config.get("source"), "config.source"),
    )
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return RawSnapshot(prices=prices, metadata=metadata)
