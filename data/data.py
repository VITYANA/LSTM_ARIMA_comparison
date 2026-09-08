"""Download and freeze the raw market-data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

DEFAULT_TICKERS = ("SPY", "AAPL", "JPM", "XOM")
DEFAULT_START = "2006-01-03"
# yfinance treats `end` as exclusive, so this includes 31 December 2025.
DEFAULT_END = "2026-01-01"
DEFAULT_INTERVAL = "1d"
DEFAULT_OUTPUT_DIR = Path("data/raw")

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
ACTION_COLUMNS = ("dividends", "stock_splits")
OUTPUT_COLUMNS = ("date", "ticker", *PRICE_COLUMNS, *ACTION_COLUMNS)


@dataclass(frozen=True)
class DownloadConfig:
    """Parameters that uniquely describe a raw-data snapshot."""

    tickers: tuple[str, ...]
    start: str
    end: str
    interval: str = DEFAULT_INTERVAL
    source: str = "Yahoo Finance via yfinance"


def normalize_ticker(ticker: str) -> str:
    """Return a normalized ticker and reject unsafe file-name characters."""
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker must not be empty")
    if any(character in normalized for character in ("/", "\\", "\0")):
        raise ValueError(f"Unsafe ticker: {ticker!r}")
    return normalized


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(package: str) -> str:
    """Return an installed package version for the reproducibility manifest."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def normalize_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance response to the project's stable long-form schema."""
    if frame.empty:
        raise ValueError(f"No observations returned for {ticker}")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        if ticker in normalized.columns.get_level_values(-1):
            normalized = cast(pd.DataFrame, normalized.xs(ticker, axis=1, level=-1))
        else:
            normalized.columns = normalized.columns.get_level_values(0)

    normalized.columns = [
        str(column).strip().lower().replace(" ", "_") for column in normalized.columns
    ]

    missing_prices = set(PRICE_COLUMNS) - set(normalized.columns)
    if missing_prices:
        raise ValueError(f"{ticker}: missing columns: {sorted(missing_prices)}")

    for column in ACTION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0.0

    dates = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    normalized.index = dates.normalize()
    normalized.index.name = "date"
    normalized = normalized.reset_index()
    normalized.insert(1, "ticker", ticker)
    return normalized.loc[:, OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)


def download_ticker(ticker: str, config: DownloadConfig) -> pd.DataFrame:
    """Download one ticker, including dividends and stock splits."""
    frame = yf.download(
        ticker,
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=False,
        actions=True,
        repair=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    return normalize_download(frame, ticker)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV representation of a data frame."""
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.10g")


def planned_output_paths(config: DownloadConfig, output_dir: Path) -> tuple[Path, ...]:
    """Return every path whose replacement must be explicitly authorized."""
    ticker_paths = tuple(output_dir / f"{ticker}.csv" for ticker in config.tickers)
    return (*ticker_paths, output_dir / "prices.csv", output_dir / "manifest.json")


def ensure_outputs_are_available(paths: Sequence[Path], force: bool) -> None:
    """Prevent an existing raw snapshot from being overwritten accidentally."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Snapshot files already exist: {rendered}. Use --force to replace.")


def file_record(path: Path, rows: int) -> dict[str, str | int]:
    """Build one file entry for the snapshot manifest."""
    return {
        "path": path.as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_snapshot(config: DownloadConfig, output_dir: Path, force: bool = False) -> Path:
    """Download all configured assets and write the raw snapshot and manifest."""
    output_paths = planned_output_paths(config, output_dir)
    ensure_outputs_are_available(output_paths, force)

    frames = [download_ticker(ticker, config) for ticker in config.tickers]
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, str | int]] = {}
    for ticker, frame in zip(config.tickers, frames, strict=True):
        path = output_dir / f"{ticker}.csv"
        write_csv(frame, path)
        files[path.name] = file_record(path, len(frame))

    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    prices_path = output_dir / "prices.csv"
    write_csv(prices, prices_path)
    files[prices_path.name] = file_record(prices_path, len(prices))

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "software": {
            "python": sys.version.split()[0],
            "pandas": package_version("pandas"),
            "yfinance": package_version("yfinance"),
        },
        "files": files,
        "notes": [
            "The end date is exclusive.",
            "auto_adjust=False; adjusted and unadjusted prices are both retained.",
            "Downloaded market data are not covered by the repository license.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END, help="Exclusive end date")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing snapshot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw-data download command."""
    args = parse_args(argv)
    tickers = tuple(dict.fromkeys(normalize_ticker(ticker) for ticker in args.tickers))
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        raise ValueError("--start must be earlier than --end")

    config = DownloadConfig(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )
    manifest_path = build_snapshot(config, args.output_dir, force=args.force)
    print(f"Snapshot created: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
