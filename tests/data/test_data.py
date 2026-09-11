"""Tests for downloading and validating market-data snapshots."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import data as data_module
from data.data import (
    ACTION_COLUMNS,
    DEFAULT_END,
    DEFAULT_INTERVAL,
    DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START,
    DEFAULT_TICKERS,
    OUTPUT_COLUMNS,
    POSITIVE_PRICE_COLUMNS,
    DownloadConfig,
    build_snapshot,
    download_ticker,
    ensure_outputs_are_available,
    file_record,
    main,
    normalize_download,
    normalize_ticker,
    package_version,
    parse_args,
    planned_output_paths,
    sha256,
    validate_ticker_frame,
    write_csv,
)


def make_raw_frame(
    rows: int = 5,
    *,
    include_actions: bool = True,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Create a deterministic frame shaped like a yfinance response"""
    dates = pd.date_range("2025-01-02", periods=rows, freq="B", tz=timezone)
    base = np.arange(rows, dtype=float) + 100.0
    values: dict[str, object] = {
        "Open": base,
        "High": base + 2.0,
        "Low": base - 2.0,
        "Close": base + 1.0,
        "Adj Close": base + 0.5,
        "Volume": np.arange(rows) * 1_000 + 1_000_000,
    }
    if include_actions:
        values["Dividends"] = np.zeros(rows)
        values["Stock Splits"] = np.zeros(rows)
    return pd.DataFrame(values, index=dates)


def make_normalized_frame(rows: int = 5, ticker: str = "SPY") -> pd.DataFrame:
    """Create a valid frame in the project's normalized schema"""
    return normalize_download(make_raw_frame(rows), ticker)


def test_download_config_defaults_and_immutability() -> None:
    """Verify default configuration values and frozen behavior"""
    config = DownloadConfig(tickers=("SPY",), start=DEFAULT_START, end=DEFAULT_END)

    assert config.interval == DEFAULT_INTERVAL
    assert config.min_observations == DEFAULT_MIN_OBSERVATIONS
    assert config.source == "Yahoo Finance via yfinance"
    with pytest.raises(FrozenInstanceError):
        config.__setattr__("start", "2020-01-01")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("spy", "SPY"),
        (" Aapl ", "AAPL"),
        ("BRK-B", "BRK-B"),
        ("^gspc", "^GSPC"),
    ],
)
def test_normalize_ticker(value: str, expected: str) -> None:
    """Normalize ticker whitespace casing and valid symbols"""
    assert normalize_ticker(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "../SPY", "A\\B", "A\0B"])
def test_normalize_ticker_rejects_invalid_values(value: str) -> None:
    """Reject empty tickers and unsafe path characters"""
    with pytest.raises(ValueError, match="Ticker must not be empty|Unsafe ticker"):
        normalize_ticker(value)


@pytest.mark.parametrize("content", [b"", b"abc", b"market-data\n" * 1000])
def test_sha256(tmp_path: Path, content: bytes) -> None:
    """Match file hashes for several byte contents"""
    path = tmp_path / "sample.bin"
    path.write_bytes(content)

    assert sha256(path) == hashlib.sha256(content).hexdigest()


def test_package_version_returns_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return installed package version from metadata provider"""

    def fake_version(package: str) -> str:
        assert package == "example"
        return "1.2.3"

    monkeypatch.setattr(data_module, "version", fake_version)
    assert package_version("example") == "1.2.3"


def test_package_version_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return unknown when package metadata is unavailable"""

    def missing_version(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr(data_module, "version", missing_version)
    assert package_version("missing") == "unknown"


@pytest.mark.parametrize(
    ("include_actions", "timezone", "reverse"),
    [(True, None, False), (False, "UTC", False), (True, None, True)],
)
def test_normalize_download_variants(
    include_actions: bool,
    timezone: str | None,
    reverse: bool,
) -> None:
    """Normalize columns dates actions and row ordering"""
    raw = make_raw_frame(include_actions=include_actions, timezone=timezone)
    if reverse:
        raw = raw.iloc[::-1]

    result = normalize_download(raw, "SPY")

    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert result["ticker"].eq("SPY").all()
    assert result["date"].is_monotonic_increasing
    assert result["date"].dt.tz is None
    if not include_actions:
        assert result.loc[:, ACTION_COLUMNS].eq(0.0).all().all()


@pytest.mark.parametrize("ticker_level", [0, 1])
def test_normalize_download_handles_multiindex(ticker_level: int) -> None:
    """Handle ticker labels on either MultiIndex level"""
    raw = make_raw_frame()
    if ticker_level == 0:
        raw.columns = pd.MultiIndex.from_tuples([("SPY", str(column)) for column in raw.columns])
    else:
        raw.columns = pd.MultiIndex.from_tuples([(str(column), "SPY") for column in raw.columns])

    result = normalize_download(raw, "SPY")

    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert len(result) == len(raw)


def test_normalize_download_rejects_different_multiindex_ticker() -> None:
    """Reject MultiIndex data belonging to another ticker"""
    raw = make_raw_frame()
    raw.columns = pd.MultiIndex.from_tuples([(str(column), "AAPL") for column in raw.columns])

    with pytest.raises(ValueError, match="Cannot identify SPY"):
        normalize_download(raw, "SPY")


def test_normalize_download_does_not_mutate_source() -> None:
    """Preserve original frame while normalizing downloaded data"""
    raw = make_raw_frame()
    original = raw.copy(deep=True)

    normalize_download(raw, "SPY")

    pd.testing.assert_frame_equal(raw, original)


def test_normalize_download_rejects_empty_frame() -> None:
    """Reject empty market data download responses"""
    with pytest.raises(ValueError, match="No observations returned"):
        normalize_download(pd.DataFrame(), "SPY")


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Adj Close", "Volume"])
def test_normalize_download_rejects_missing_price_column(column: str) -> None:
    """Reject downloads missing required market price columns"""
    raw = make_raw_frame().drop(columns=column)

    with pytest.raises(ValueError, match="missing columns"):
        normalize_download(raw, "SPY")


@pytest.mark.parametrize("rows", [1, 5, 100])
def test_validate_ticker_frame_accepts_valid_frames(rows: int) -> None:
    """Accept valid frames at several minimum lengths"""
    frame = make_normalized_frame(rows)
    validate_ticker_frame(frame, "SPY", min_observations=rows)


@pytest.mark.parametrize("minimum", [0, -1, -100])
def test_validate_ticker_frame_rejects_invalid_minimum(minimum: int) -> None:
    """Reject nonpositive minimum observation requirements"""
    with pytest.raises(ValueError, match="min_observations must be positive"):
        validate_ticker_frame(make_normalized_frame(), "SPY", minimum)


@pytest.mark.parametrize("column", OUTPUT_COLUMNS)
def test_validate_ticker_frame_rejects_missing_columns(column: str) -> None:
    """Reject every missing required normalized column"""
    frame = make_normalized_frame().drop(columns=column)

    with pytest.raises(ValueError, match="missing columns"):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("short", "only 3 observations"),
        ("missing_date", "missing dates"),
        ("unsorted", "dates are not sorted"),
        ("duplicate_date", "duplicate dates"),
        ("wrong_ticker", "ticker column contains unexpected values"),
        ("missing_ticker", "ticker column contains unexpected values"),
    ],
)
def test_validate_ticker_frame_rejects_structural_errors(case: str, message: str) -> None:
    """Reject invalid dates ordering duplicates and ticker values"""
    frame = make_normalized_frame(3)
    minimum = 1
    if case == "short":
        minimum = 4
    elif case == "missing_date":
        frame.loc[1, "date"] = pd.NaT
    elif case == "unsorted":
        frame = frame.iloc[[1, 0, 2]].reset_index(drop=True)
    elif case == "duplicate_date":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif case == "wrong_ticker":
        frame.loc[1, "ticker"] = "AAPL"
    elif case == "missing_ticker":
        frame.loc[1, "ticker"] = None

    with pytest.raises(ValueError, match=message):
        validate_ticker_frame(frame, "SPY", min_observations=minimum)


@pytest.mark.parametrize("column", (*data_module.PRICE_COLUMNS, *ACTION_COLUMNS))
def test_validate_ticker_frame_rejects_missing_numeric_values(column: str) -> None:
    """Reject missing values across all numeric columns"""
    frame = make_normalized_frame()
    frame.loc[0, column] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("close", np.inf, "non-finite market values"),
        ("volume", -1, "negative volume"),
        ("dividends", -0.1, "negative dividend or split"),
        ("stock_splits", -1.0, "negative dividend or split"),
    ],
)
def test_validate_ticker_frame_rejects_invalid_numeric_values(
    column: str,
    value: float,
    message: str,
) -> None:
    """Reject infinity negatives and invalid corporate actions"""
    frame = make_normalized_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize("column", (*data_module.PRICE_COLUMNS, *ACTION_COLUMNS))
def test_validate_ticker_frame_rejects_non_numeric_values(column: str) -> None:
    """Reject text values in every numeric column"""
    frame = make_normalized_frame()
    values = frame[column].astype(object)
    values.iloc[0] = "not-a-number"
    frame[column] = values

    with pytest.raises(ValueError, match="non-numeric market values"):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize("column", (*data_module.PRICE_COLUMNS, *ACTION_COLUMNS))
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_validate_ticker_frame_rejects_all_non_finite_values(
    column: str,
    value: float,
) -> None:
    """Reject positive and negative infinity across columns"""
    frame = make_normalized_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match="non-finite market values"):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize("column", POSITIVE_PRICE_COLUMNS)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_validate_ticker_frame_rejects_non_positive_prices(column: str, value: float) -> None:
    """Reject zero and negative values in price columns"""
    frame = make_normalized_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match="non-positive prices"):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize(
    ("column", "reference", "message"),
    [
        ("high", "open", "high is below"),
        ("high", "close", "high is below"),
        ("low", "open", "low is above"),
        ("low", "close", "low is above"),
    ],
)
def test_validate_ticker_frame_rejects_inconsistent_ohlc(
    column: str,
    reference: str,
    message: str,
) -> None:
    """Reject high and low values outside OHLC bounds"""
    frame = make_normalized_frame()
    offset = -1.0 if column == "high" else 1.0
    reference_value = frame[reference].to_numpy(dtype=float)[0]
    frame.loc[0, column] = reference_value + offset

    with pytest.raises(ValueError, match=message):
        validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize(
    ("column", "reference", "offset"),
    [
        ("high", "close", -data_module.TOLERANCE / 2),
        ("low", "open", data_module.TOLERANCE / 2),
    ],
)
def test_validate_ticker_frame_accepts_rounding_within_tolerance(
    column: str,
    reference: str,
    offset: float,
) -> None:
    """Allow insignificant floating-point deviations within OHLC tolerance"""
    frame = make_normalized_frame()
    reference_value = frame[reference].to_numpy(dtype=float)[0]
    frame.loc[0, column] = reference_value + offset

    validate_ticker_frame(frame, "SPY", min_observations=1)


def test_validate_ticker_frame_accepts_zero_volume_and_actions() -> None:
    """Accept zero volume dividends and split values"""
    frame = make_normalized_frame()
    frame.loc[0, "volume"] = 0
    frame.loc[0, ["dividends", "stock_splits"]] = 0.0

    validate_ticker_frame(frame, "SPY", min_observations=1)


@pytest.mark.parametrize("ticker", ["SPY", "AAPL"])
def test_download_ticker_passes_expected_options(
    monkeypatch: pytest.MonkeyPatch,
    ticker: str,
) -> None:
    """Pass reproducible options into mocked yfinance downloads"""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_download(symbol: str, **options: object) -> pd.DataFrame:
        calls.append((symbol, options))
        return make_raw_frame(3, include_actions=False)

    monkeypatch.setattr("data.data.yf.download", fake_download)
    config = DownloadConfig(
        tickers=(ticker,),
        start="2025-01-02",
        end="2025-01-10",
        min_observations=3,
    )

    result = download_ticker(ticker, config)

    assert len(result) == 3
    assert calls == [
        (
            ticker,
            {
                "start": "2025-01-02",
                "end": "2025-01-10",
                "interval": "1d",
                "auto_adjust": False,
                "actions": True,
                "repair": True,
                "progress": False,
                "threads": False,
                "multi_level_index": False,
            },
        )
    ]


def test_download_ticker_validates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate normalized response before returning downloaded ticker"""
    monkeypatch.setattr(
        "data.data.yf.download",
        lambda *_args, **_kwargs: make_raw_frame(2),
    )
    config = DownloadConfig(
        tickers=("SPY",),
        start="2025-01-02",
        end="2025-01-10",
        min_observations=3,
    )

    with pytest.raises(ValueError, match="only 2 observations"):
        download_ticker("SPY", config)


@pytest.mark.parametrize("rows", [1, 3])
def test_write_csv_uses_stable_format(tmp_path: Path, rows: int) -> None:
    """Write expected CSV header dates and row counts"""
    path = tmp_path / "prices.csv"
    write_csv(make_normalized_frame(rows), path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("date,ticker,open,high,low,close,adj_close,volume")
    assert "2025-01-02" in text
    assert len(text.splitlines()) == rows + 1


def test_write_csv_is_deterministic(tmp_path: Path) -> None:
    """Produce byte-identical CSV files from identical frames"""
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    frame = make_normalized_frame()

    write_csv(frame, first_path)
    write_csv(frame, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert sha256(first_path) == sha256(second_path)


@pytest.mark.parametrize(
    ("tickers", "expected_names"),
    [
        (("SPY",), ("SPY.csv", "prices.csv", "manifest.json")),
        (("SPY", "AAPL"), ("SPY.csv", "AAPL.csv", "prices.csv", "manifest.json")),
    ],
)
def test_planned_output_paths(
    tmp_path: Path,
    tickers: tuple[str, ...],
    expected_names: tuple[str, ...],
) -> None:
    """Generate expected output paths for configured tickers"""
    config = DownloadConfig(tickers=tickers, start=DEFAULT_START, end=DEFAULT_END)

    paths = planned_output_paths(config, tmp_path)

    assert tuple(path.name for path in paths) == expected_names
    assert all(path.parent == tmp_path for path in paths)


@pytest.mark.parametrize(
    ("file_exists", "force", "should_raise"),
    [(False, False, False), (False, True, False), (True, True, False), (True, False, True)],
)
def test_ensure_outputs_are_available(
    tmp_path: Path,
    file_exists: bool,
    force: bool,
    should_raise: bool,
) -> None:
    """Handle existing outputs according to force setting"""
    path = tmp_path / "prices.csv"
    if file_exists:
        path.write_text("existing", encoding="utf-8")

    if should_raise:
        with pytest.raises(FileExistsError, match="Use --force"):
            ensure_outputs_are_available((path,), force)
    else:
        ensure_outputs_are_available((path,), force)


def test_ensure_outputs_are_available_lists_every_existing_path(tmp_path: Path) -> None:
    """Report every conflicting path in overwrite error"""
    first = tmp_path / "SPY.csv"
    second = tmp_path / "prices.csv"
    first.touch()
    second.touch()

    with pytest.raises(FileExistsError) as error:
        ensure_outputs_are_available((first, second), force=False)

    assert str(first) in str(error.value)
    assert str(second) in str(error.value)


@pytest.mark.parametrize(("content", "rows"), [(b"abc", 1), (b"prices\n1\n2\n", 2)])
def test_file_record(tmp_path: Path, content: bytes, rows: int) -> None:
    """Build accurate manifest metadata for different files"""
    path = tmp_path / "file.csv"
    path.write_bytes(content)

    record = file_record(path, rows)

    assert record == {
        "path": path.as_posix(),
        "rows": rows,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_sha256_and_file_record_reject_missing_file(tmp_path: Path) -> None:
    """Raise when hashing or recording nonexistent files"""
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        sha256(missing)
    with pytest.raises(FileNotFoundError):
        file_record(missing, rows=0)


@pytest.mark.parametrize("tickers", [("SPY",), ("SPY", "AAPL")])
def test_build_snapshot_creates_files_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tickers: tuple[str, ...],
) -> None:
    """Create per-ticker combined files and complete manifest"""

    def fake_download(ticker: str, _config: DownloadConfig) -> pd.DataFrame:
        return make_normalized_frame(3, ticker)

    monkeypatch.setattr(data_module, "download_ticker", fake_download)
    config = DownloadConfig(
        tickers=tickers,
        start="2025-01-02",
        end="2025-01-10",
        min_observations=3,
    )
    output_dir = tmp_path / "raw"

    manifest_path = build_snapshot(config, output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined = pd.read_csv(output_dir / "prices.csv")

    assert manifest["config"]["tickers"] == list(tickers)
    assert manifest["config"]["min_observations"] == 3
    assert manifest["software"]["python"] == sys.version.split()[0]
    assert datetime.fromisoformat(manifest["created_at_utc"]).tzinfo is not None
    assert set(manifest["files"]) == {*[f"{ticker}.csv" for ticker in tickers], "prices.csv"}
    assert len(combined) == 3 * len(tickers)
    assert combined.sort_values(["ticker", "date"]).reset_index(drop=True).equals(combined)
    for ticker in tickers:
        ticker_path = output_dir / f"{ticker}.csv"
        record = manifest["files"][f"{ticker}.csv"]
        assert ticker_path.is_file()
        assert record["rows"] == 3
        assert record["bytes"] == ticker_path.stat().st_size
        assert record["sha256"] == sha256(ticker_path)


def test_build_snapshot_checks_overwrite_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block overwrite before initiating any network download"""
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    (output_dir / "SPY.csv").write_text("existing", encoding="utf-8")

    def unexpected_download(_ticker: str, _config: DownloadConfig) -> pd.DataFrame:
        pytest.fail("download_ticker must not be called")

    monkeypatch.setattr(data_module, "download_ticker", unexpected_download)
    config = DownloadConfig(tickers=("SPY",), start=DEFAULT_START, end=DEFAULT_END)

    with pytest.raises(FileExistsError, match="Use --force"):
        build_snapshot(config, output_dir)


def test_build_snapshot_does_not_write_after_download_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid partial files when one download fails"""
    output_dir = tmp_path / "raw"

    def failing_download(ticker: str, _config: DownloadConfig) -> pd.DataFrame:
        if ticker == "AAPL":
            raise ConnectionError("network failure")
        return make_normalized_frame(3, ticker)

    monkeypatch.setattr(data_module, "download_ticker", failing_download)
    config = DownloadConfig(
        tickers=("SPY", "AAPL"),
        start=DEFAULT_START,
        end=DEFAULT_END,
    )

    with pytest.raises(ConnectionError, match="network failure"):
        build_snapshot(config, output_dir)
    assert not output_dir.exists()


def test_build_snapshot_force_replaces_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace existing snapshot files when force enabled"""
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    old_path = output_dir / "SPY.csv"
    old_path.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        data_module,
        "download_ticker",
        lambda ticker, _config: make_normalized_frame(3, ticker),
    )
    config = DownloadConfig(tickers=("SPY",), start=DEFAULT_START, end=DEFAULT_END)

    build_snapshot(config, output_dir, force=True)

    assert old_path.read_text(encoding="utf-8").startswith("date,ticker")


def test_parse_args_uses_defaults() -> None:
    """Return documented defaults for every CLI argument"""
    args = parse_args([])

    assert args.tickers == list(DEFAULT_TICKERS)
    assert args.start == DEFAULT_START
    assert args.end == DEFAULT_END
    assert args.interval == DEFAULT_INTERVAL
    assert args.min_observations == DEFAULT_MIN_OBSERVATIONS
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.force is False


@pytest.mark.parametrize(
    ("arguments", "attribute", "expected"),
    [
        (["--tickers", "SPY", "AAPL"], "tickers", ["SPY", "AAPL"]),
        (["--interval", "1wk"], "interval", "1wk"),
        (["--min-observations", "25"], "min_observations", 25),
        (["--output-dir", "custom/raw"], "output_dir", Path("custom/raw")),
        (["--force"], "force", True),
    ],
)
def test_parse_args_custom_values(arguments: list[str], attribute: str, expected: object) -> None:
    """Parse each supported custom command-line option"""
    assert getattr(parse_args(arguments), attribute) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tickers"],
        ["--min-observations", "not-an-integer"],
        ["--unknown-option"],
    ],
)
def test_parse_args_rejects_malformed_arguments(arguments: list[str]) -> None:
    """Reject missing malformed and unknown CLI arguments"""
    with pytest.raises(SystemExit) as error:
        parse_args(arguments)

    assert error.value.code == 2


def test_main_normalizes_tickers_and_builds_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Normalize CLI tickers and forward snapshot options"""
    captured: dict[str, object] = {}
    expected_manifest = tmp_path / "manifest.json"

    def fake_build(config: DownloadConfig, output_dir: Path, force: bool = False) -> Path:
        captured.update(config=config, output_dir=output_dir, force=force)
        return expected_manifest

    monkeypatch.setattr(data_module, "build_snapshot", fake_build)

    result = main(
        [
            "--tickers",
            " spy ",
            "AAPL",
            "SPY",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-10",
            "--min-observations",
            "3",
            "--output-dir",
            str(tmp_path),
            "--force",
        ]
    )

    config = captured["config"]
    assert isinstance(config, DownloadConfig)
    assert config.tickers == ("SPY", "AAPL")
    assert config.min_observations == 3
    assert captured["output_dir"] == tmp_path
    assert captured["force"] is True
    assert result == 0
    assert str(expected_manifest) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--start", "2025-01-10", "--end", "2025-01-01"], "start must be earlier"),
        (["--min-observations", "0"], "min-observations must be positive"),
        (["--tickers", "../SPY"], "Unsafe ticker"),
    ],
)
def test_main_rejects_invalid_arguments(arguments: list[str], message: str) -> None:
    """Reject invalid dates observation limits and tickers"""
    with pytest.raises(ValueError, match=message):
        main(arguments)


def test_module_entrypoint_displays_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute module entrypoint and display CLI help"""
    monkeypatch.setattr(sys, "argv", ["data.data", "--help"])

    with (
        pytest.warns(RuntimeWarning, match="found in sys.modules"),
        pytest.raises(SystemExit) as error,
    ):
        runpy.run_module("data.data", run_name="__main__")

    assert error.value.code == 0
    assert "Download and freeze" in capsys.readouterr().out
