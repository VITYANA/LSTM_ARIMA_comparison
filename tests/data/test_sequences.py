"""Tests for leakage-safe univariate return sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.sequences import (
    SequenceBatch,
    build_return_sequences,
    fit_return_scaler,
    inverse_transform_targets,
    lag_columns,
    split_sequence_batch,
    transform_features,
    transform_targets,
)

TRAIN_ROWS = 65
VALIDATION_ROWS = 5
TEST_ROWS = 3


def make_dataset(
    *,
    train_rows: int = TRAIN_ROWS,
    validation_rows: int = VALIDATION_ROWS,
    test_rows: int = TEST_ROWS,
) -> pd.DataFrame:
    """Build deterministic model rows spanning every research split"""
    row_count = train_rows + validation_rows + test_rows
    dates = pd.bdate_range("2019-01-01", periods=row_count + 1)
    returns = np.arange(1, row_count + 1, dtype="float64") / 1_000
    splits = [
        *(["train"] * train_rows),
        *(["validation"] * validation_rows),
        *(["test"] * test_rows),
    ]
    return pd.DataFrame(
        {
            "date": dates[:-1],
            "target_date": dates[1:],
            "ticker": "AAA",
            "adj_close": 100.0 + np.arange(row_count),
            "log_return": returns,
            "target_return": returns + 0.0005,
            "split": splits,
        },
        index=pd.Index(np.arange(100, 100 + row_count), name="source_index"),
    )


@pytest.mark.parametrize("window", [5, 21, 63])
def test_build_return_sequences_aligns_windows(window: int) -> None:
    """Align oldest-to-current lags with validation targets"""
    dataset = make_dataset()
    first_validation_index = dataset.index[TRAIN_ROWS]
    first_validation_position = TRAIN_ROWS
    expected_lags = (
        dataset["log_return"]
        .iloc[first_validation_position - window + 1 : first_validation_position + 1]
        .to_numpy()
    )

    batch = build_return_sequences(dataset, "AAA", window, "validation")

    assert batch.features.columns.tolist() == list(lag_columns(window))
    assert batch.features.index.equals(batch.targets.index)
    assert batch.features.index.equals(batch.keys.index)
    assert batch.features.index[0] == first_validation_index
    assert batch.features.iloc[0].to_numpy() == pytest.approx(expected_lags)
    assert batch.targets.iloc[0] == pytest.approx(
        dataset.loc[first_validation_index, "target_return"]
    )
    assert batch.keys.iloc[0].to_dict() == {
        "date": dataset.loc[first_validation_index, "date"],
        "target_date": dataset.loc[first_validation_index, "target_date"],
        "ticker": "AAA",
        "split": "validation",
    }


def test_build_return_sequences_sorts_without_losing_source_indexes() -> None:
    """Normalize shuffled rows while retaining target indexes"""
    dataset = make_dataset()
    shuffled = dataset.sample(frac=1, random_state=17)
    expected_indexes = dataset.index[dataset["split"].eq("validation")]

    batch = build_return_sequences(shuffled, "AAA", 5, "validation")

    assert batch.keys["target_date"].is_monotonic_increasing
    assert batch.keys.index.tolist() == expected_indexes.tolist()


def test_build_return_sequences_excludes_final_test_values() -> None:
    """Keep final test returns outside validation windows"""
    dataset = make_dataset()
    dataset.loc[dataset["split"].eq("test"), "log_return"] = 999.0

    batch = build_return_sequences(dataset, "AAA", 5, "validation")

    assert batch.keys["split"].eq("validation").all()
    assert not np.isclose(batch.features.to_numpy(), 999.0).any()


def test_build_return_sequences_does_not_mutate_source() -> None:
    """Preserve source rows columns indexes and ordering"""
    dataset = make_dataset().sample(frac=1, random_state=11)
    original = dataset.copy(deep=True)

    build_return_sequences(dataset, "AAA", 5, "validation")

    pd.testing.assert_frame_equal(dataset, original)


@pytest.mark.parametrize("invalid_window", [True, 0, -1, 2.5, "5"])
def test_lag_columns_rejects_invalid_windows(invalid_window: object) -> None:
    """Reject noninteger and nonpositive window lengths"""
    with pytest.raises(ValueError, match="window"):
        lag_columns(invalid_window)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_ticker", [None, "", "   ", 7])
def test_build_return_sequences_rejects_invalid_tickers(invalid_ticker: object) -> None:
    """Reject missing blank and nonstring ticker names"""
    with pytest.raises(ValueError, match="ticker"):
        build_return_sequences(
            make_dataset(),
            invalid_ticker,  # type: ignore[arg-type]
            5,
            "validation",
        )


def test_build_return_sequences_rejects_unknown_ticker() -> None:
    """Reject ticker absent from the verified dataset"""
    with pytest.raises(ValueError, match="ticker.*absent"):
        build_return_sequences(make_dataset(), "BBB", 5, "validation")


@pytest.mark.parametrize("target_split", ["test", "holdout", "", None])
def test_build_return_sequences_rejects_invalid_target_splits(
    target_split: str | None,
) -> None:
    """Allow only train and validation sequence targets"""
    with pytest.raises(ValueError, match="target_split"):
        build_return_sequences(
            make_dataset(),
            "AAA",
            5,
            target_split,  # type: ignore[arg-type]
        )


def test_build_return_sequences_rejects_missing_columns() -> None:
    """Reject datasets missing required model columns"""
    invalid = make_dataset().drop(columns="log_return")

    with pytest.raises(ValueError, match="missing columns.*log_return"):
        build_return_sequences(invalid, "AAA", 5, "validation")


def test_build_return_sequences_rejects_duplicate_keys() -> None:
    """Reject duplicate ticker target and split keys"""
    dataset = make_dataset()
    invalid = pd.concat([dataset, dataset.iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate"):
        build_return_sequences(invalid, "AAA", 5, "validation")


@pytest.mark.parametrize("invalid_return", ["invalid", np.nan, np.inf, -np.inf])
def test_build_return_sequences_rejects_invalid_returns(invalid_return: object) -> None:
    """Reject nonnumeric and nonfinite modeling returns"""
    invalid = make_dataset()
    if isinstance(invalid_return, str):
        invalid["log_return"] = invalid["log_return"].astype(object)
    invalid.loc[invalid.index[0], "log_return"] = invalid_return  # type: ignore[call-overload]

    with pytest.raises(ValueError, match="log_return"):
        build_return_sequences(invalid, "AAA", 5, "validation")


def test_build_return_sequences_rejects_insufficient_history() -> None:
    """Reject target split lacking one complete input window"""
    dataset = make_dataset(train_rows=2, validation_rows=1, test_rows=1)

    with pytest.raises(ValueError, match="insufficient"):
        build_return_sequences(dataset, "AAA", 5, "validation")


def test_build_return_sequences_rejects_missing_target_split_rows() -> None:
    """Reject datasets without requested target split rows"""
    dataset = make_dataset(validation_rows=0)

    with pytest.raises(ValueError, match="validation.*rows"):
        build_return_sequences(dataset, "AAA", 5, "validation")


def test_split_sequence_batch_uses_chronological_fraction() -> None:
    """Split batches chronologically without overlapping indexes"""
    batch = build_return_sequences(make_dataset(), "AAA", 5, "train")
    original_features = batch.features.copy(deep=True)

    inner_train, inner_validation = split_sequence_batch(batch)

    expected_train_rows = int(len(batch.features) * 0.8)
    assert len(inner_train.features) == expected_train_rows
    assert len(inner_validation.features) == len(batch.features) - expected_train_rows
    assert inner_train.keys["target_date"].max() < inner_validation.keys["target_date"].min()
    assert set(inner_train.features.index).isdisjoint(inner_validation.features.index)
    pd.testing.assert_frame_equal(batch.features, original_features)


@pytest.mark.parametrize("invalid_fraction", [True, 0.0, 1.0, -0.1, 1.1, "0.8"])
def test_split_sequence_batch_rejects_invalid_fractions(invalid_fraction: object) -> None:
    """Reject fractions that cannot form ordered partitions"""
    batch = build_return_sequences(make_dataset(), "AAA", 5, "train")

    with pytest.raises(ValueError, match="train_fraction"):
        split_sequence_batch(batch, invalid_fraction)  # type: ignore[arg-type]


def test_split_sequence_batch_rejects_empty_partition() -> None:
    """Reject batches too short for both inner partitions"""
    index = pd.Index([10])
    batch = SequenceBatch(
        features=pd.DataFrame({"lag_0": [0.01]}, index=index),
        targets=pd.Series([0.02], index=index, name="target_return"),
        keys=pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")],
                "target_date": [pd.Timestamp("2020-01-02")],
                "ticker": ["AAA"],
                "split": ["train"],
            },
            index=index,
        ),
    )

    with pytest.raises(ValueError, match="partitions"):
        split_sequence_batch(batch)


@pytest.mark.parametrize("mismatch", ["target_length", "target_index", "keys_index"])
def test_split_sequence_batch_rejects_misaligned_components(mismatch: str) -> None:
    """Reject unequal lengths and indexes across batch components"""
    batch = build_return_sequences(make_dataset(), "AAA", 5, "train")
    targets = batch.targets.copy()
    keys = batch.keys.copy()
    if mismatch == "target_length":
        targets = targets.iloc[:-1]
    elif mismatch == "target_index":
        targets.index = targets.index + 1_000
    else:
        keys.index = keys.index + 1_000
    invalid = SequenceBatch(features=batch.features, targets=targets, keys=keys)

    with pytest.raises(ValueError, match="aligned"):
        split_sequence_batch(invalid)


def test_fit_return_scaler_uses_only_history_through_cutoff() -> None:
    """Estimate scaler strictly through the supplied origin date"""
    dataset = make_dataset()
    cutoff_index = dataset.index[9]
    cutoff_date = dataset.loc[cutoff_index, "date"]
    expected = dataset.loc[dataset["date"].le(cutoff_date), "log_return"]

    scaler = fit_return_scaler(dataset, "AAA", cutoff_date)

    assert scaler.mean_[0] == pytest.approx(expected.mean())
    assert scaler.var_[0] == pytest.approx(expected.var(ddof=0))


def test_fit_return_scaler_excludes_final_test_rows() -> None:
    """Exclude final test values even with a later cutoff"""
    dataset = make_dataset()
    evaluation = dataset.loc[dataset["split"].ne("test"), "log_return"]
    dataset.loc[dataset["split"].eq("test"), "log_return"] = 999.0

    scaler = fit_return_scaler(dataset, "AAA", dataset["date"].max())

    assert scaler.mean_[0] == pytest.approx(evaluation.mean())


@pytest.mark.parametrize(
    ("ticker", "through_date", "message"),
    [
        ("BBB", pd.Timestamp("2020-01-01"), "ticker.*absent"),
        ("AAA", pd.Timestamp("1900-01-01"), "history"),
    ],
)
def test_fit_return_scaler_rejects_empty_histories(
    ticker: str,
    through_date: pd.Timestamp,
    message: str,
) -> None:
    """Reject unknown tickers and cutoffs before history"""
    with pytest.raises(ValueError, match=message):
        fit_return_scaler(make_dataset(), ticker, through_date)


def test_scaler_helpers_transform_and_restore_returns() -> None:
    """Transform feature tensors and restore target scale"""
    dataset = make_dataset()
    batch = build_return_sequences(dataset, "AAA", 5, "train")
    scaler = fit_return_scaler(dataset, "AAA", batch.keys["date"].max())
    original_features = batch.features.copy(deep=True)
    original_targets = batch.targets.copy(deep=True)

    features = transform_features(batch.features, scaler)
    targets = transform_targets(batch.targets, scaler)
    restored = inverse_transform_targets(targets.reshape(-1, 1), scaler)

    assert features.shape == (len(batch.features), 5, 1)
    assert features.dtype == np.float64
    assert targets.shape == (len(batch.targets),)
    assert targets.dtype == np.float64
    assert restored == pytest.approx(batch.targets.to_numpy())
    pd.testing.assert_frame_equal(batch.features, original_features)
    pd.testing.assert_series_equal(batch.targets, original_targets)


def test_scaler_helpers_keep_constant_returns_finite() -> None:
    """Keep zero-variance train transformations finite"""
    dataset = make_dataset()
    dataset["log_return"] = 0.01
    dataset["target_return"] = 0.01
    batch = build_return_sequences(dataset, "AAA", 5, "train")
    scaler = fit_return_scaler(dataset, "AAA", batch.keys["date"].max())

    features = transform_features(batch.features, scaler)
    targets = transform_targets(batch.targets, scaler)
    restored = inverse_transform_targets(targets, scaler)

    assert np.isfinite(features).all()
    assert np.isfinite(targets).all()
    assert restored == pytest.approx(np.full(len(targets), 0.01))


@pytest.mark.parametrize("column", ["features", "targets"])
@pytest.mark.parametrize("invalid_value", ["invalid", np.nan, np.inf, -np.inf])
def test_scaler_helpers_reject_invalid_values(column: str, invalid_value: object) -> None:
    """Reject invalid values before scaler transformations"""
    dataset = make_dataset()
    batch = build_return_sequences(dataset, "AAA", 5, "train")
    scaler = fit_return_scaler(dataset, "AAA", batch.keys["date"].max())

    if column == "features":
        invalid_features = batch.features.copy()
        if isinstance(invalid_value, str):
            invalid_features["lag_0"] = invalid_features["lag_0"].astype(object)
        invalid_features.iloc[0, -1] = invalid_value  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="features"):
            transform_features(invalid_features, scaler)
    else:
        invalid_targets = batch.targets.copy()
        if isinstance(invalid_value, str):
            invalid_targets = invalid_targets.astype(object)
        invalid_targets.iloc[0] = invalid_value
        with pytest.raises(ValueError, match="targets"):
            transform_targets(invalid_targets, scaler)


@pytest.mark.parametrize(
    "invalid_values",
    [
        np.array([[1.0, 2.0]]),
        np.array([[[1.0]]]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_inverse_transform_targets_rejects_invalid_shapes_and_values(
    invalid_values: np.ndarray,
) -> None:
    """Reject nonvector and nonfinite inverse inputs"""
    dataset = make_dataset()
    scaler = fit_return_scaler(dataset, "AAA", dataset.loc[dataset.index[9], "date"])

    with pytest.raises(ValueError, match="values"):
        inverse_transform_targets(invalid_values, scaler)
