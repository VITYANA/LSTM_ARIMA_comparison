"""Tests for leakage-safe model-dataset preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.preparation import SplitConfig, assign_time_split, build_model_dataset

TEST_SPLITS = SplitConfig(
    train_start="2019-12-30",
    train_end="2019-12-31",
    validation_start="2020-01-01",
    validation_end="2020-01-02",
    test_start="2020-01-03",
    test_end="2020-01-06",
)


def prices_for_two_tickers() -> pd.DataFrame:
    """Return shuffled prices spanning all split boundaries"""
    dates = pd.to_datetime(
        [
            "2019-12-30",
            "2019-12-31",
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
        ]
    )
    frame = pd.DataFrame(
        {
            "date": [*dates, *dates],
            "ticker": ["AAA"] * 6 + ["BBB"] * 6,
            "adj_close": [
                100.0,
                101.0,
                102.0,
                104.0,
                103.0,
                105.0,
                50.0,
                51.0,
                50.0,
                52.0,
                54.0,
                53.0,
            ],
        }
    )
    return frame.sample(frac=1, random_state=5).reset_index(drop=True)


def test_build_model_dataset_shifts_target_within_ticker() -> None:
    """Build next-day targets without crossing ticker boundaries"""
    result = build_model_dataset(prices_for_two_tickers(), TEST_SPLITS)
    aaa = result[result["ticker"].eq("AAA")].reset_index(drop=True)

    assert result.columns.tolist() == [
        "date",
        "target_date",
        "ticker",
        "adj_close",
        "log_return",
        "target_return",
        "split",
    ]
    assert aaa.loc[0, "date"] == pd.Timestamp("2019-12-31")
    assert aaa.loc[0, "target_date"] == pd.Timestamp("2020-01-01")
    assert aaa.loc[0, "target_return"] == pytest.approx(np.log(102 / 101))
    assert aaa.loc[0, "split"] == "validation"
    assert result.groupby("ticker").size().to_dict() == {"AAA": 4, "BBB": 4}


@pytest.mark.parametrize(
    ("target_date", "expected"),
    [
        ("2019-12-30", "train"),
        ("2019-12-31", "train"),
        ("2020-01-01", "validation"),
        ("2020-01-02", "validation"),
        ("2020-01-03", "test"),
        ("2020-01-06", "test"),
    ],
)
def test_assign_time_split_uses_target_date(target_date: str, expected: str) -> None:
    """Assign exact boundaries from forecast target dates"""
    frame = pd.DataFrame({"target_date": [target_date]})

    result = assign_time_split(frame, TEST_SPLITS)

    assert result.loc[0, "split"] == expected
    assert result.loc[0, "target_date"] == pd.Timestamp(target_date)


def test_assign_time_split_rejects_uncovered_dates() -> None:
    """Reject target dates outside configured research periods"""
    frame = pd.DataFrame({"target_date": ["2020-01-07"]})

    with pytest.raises(ValueError, match="outside configured splits"):
        assign_time_split(frame, TEST_SPLITS)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(
            SplitConfig(
                train_start="2020-01-01",
                train_end="2020-01-03",
                validation_start="2020-01-03",
                validation_end="2020-01-04",
                test_start="2020-01-05",
                test_end="2020-01-06",
            ),
            id="overlapping",
        ),
        pytest.param(
            SplitConfig(
                train_start="2020-01-01",
                train_end="2020-01-04",
                validation_start="2020-01-03",
                validation_end="2020-01-05",
                test_start="2020-01-06",
                test_end="2020-01-07",
            ),
            id="reversed",
        ),
    ],
)
def test_assign_time_split_rejects_invalid_boundaries(invalid: SplitConfig) -> None:
    """Reject overlapping or reversed chronological periods"""
    frame = pd.DataFrame({"target_date": ["2020-01-02"]})

    with pytest.raises(ValueError, match="strictly increasing"):
        assign_time_split(frame, invalid)


def test_assign_time_split_does_not_mutate_source() -> None:
    """Preserve source dates columns and row ordering"""
    frame = pd.DataFrame({"target_date": ["2019-12-31"]})
    original = frame.copy(deep=True)

    assign_time_split(frame, TEST_SPLITS)

    pd.testing.assert_frame_equal(frame, original)


def test_build_model_dataset_rejects_infinite_returns() -> None:
    """Reject non-finite features before model consumption"""
    prices = prices_for_two_tickers()
    prices.loc[prices["ticker"].eq("AAA"), "adj_close"] = [
        100,
        101,
        np.inf,
        104,
        103,
        105,
    ]

    with pytest.raises(ValueError, match="finite"):
        build_model_dataset(prices, TEST_SPLITS)


def test_build_model_dataset_does_not_mutate_source() -> None:
    """Leave the verified raw frame unchanged"""
    prices = prices_for_two_tickers()
    original = prices.copy(deep=True)

    build_model_dataset(prices, TEST_SPLITS)

    pd.testing.assert_frame_equal(prices, original)
