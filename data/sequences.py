"""Build leakage-safe univariate return sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from evaluation.contracts import (
    EVALUATION_SPLITS,
    TARGET_COLUMN,
    require_finite_numeric,
    validate_model_dataset,
)

KEY_COLUMNS = ("date", "target_date", "ticker", "split")
RETURN_COLUMN = "log_return"
VALID_TARGET_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class SequenceBatch:
    """Contain aligned lag features, targets and observation keys."""

    features: pd.DataFrame
    targets: pd.Series
    keys: pd.DataFrame


def _validate_window(window: object) -> int:
    """Return a positive integer window length."""
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    return window


def _validate_ticker(ticker: object) -> str:
    """Return a non-empty ticker identifier."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    return ticker


def lag_columns(window: int) -> tuple[str, ...]:
    """Return lag columns ordered from oldest to current return."""
    validated_window = _validate_window(window)
    return tuple(f"lag_{lag}" for lag in range(validated_window - 1, -1, -1))


def build_return_sequences(
    dataset: pd.DataFrame,
    ticker: str,
    window: int,
    target_split: str,
) -> SequenceBatch:
    """Build aligned return windows for one ticker and target split."""
    validate_model_dataset(dataset)
    validated_ticker = _validate_ticker(ticker)
    columns = lag_columns(window)
    if target_split not in VALID_TARGET_SPLITS:
        raise ValueError("target_split must be 'train' or 'validation'")
    if not dataset["ticker"].eq(validated_ticker).any():
        raise ValueError(f"ticker {validated_ticker!r} is absent from dataset")

    rows = dataset.loc[
        dataset["ticker"].eq(validated_ticker) & dataset["split"].isin(EVALUATION_SPLITS)
    ].copy()
    rows = rows.sort_values(["date", "target_date"])
    require_finite_numeric(rows[RETURN_COLUMN], RETURN_COLUMN)
    if not rows["split"].eq(target_split).any():
        raise ValueError(f"target split {target_split!r} has no rows")

    features = pd.DataFrame(
        {
            column: rows[RETURN_COLUMN].shift(lag)
            for column, lag in zip(columns, range(len(columns) - 1, -1, -1), strict=True)
        },
        index=rows.index,
    )
    selected = rows["split"].eq(target_split) & features.notna().all(axis=1)
    if not selected.any():
        raise ValueError(
            f"insufficient history for {target_split!r} sequences with window {window}"
        )

    selected_features = features.loc[selected].astype("float64")
    targets = rows.loc[selected, TARGET_COLUMN].astype("float64").copy()
    targets.name = TARGET_COLUMN
    keys = rows.loc[selected, KEY_COLUMNS].copy()
    return SequenceBatch(features=selected_features, targets=targets, keys=keys)


def _slice_batch(batch: SequenceBatch, positions: slice) -> SequenceBatch:
    """Return a copied positional slice of one sequence batch."""
    return SequenceBatch(
        features=batch.features.iloc[positions].copy(),
        targets=batch.targets.iloc[positions].copy(),
        keys=batch.keys.iloc[positions].copy(),
    )


def split_sequence_batch(
    batch: SequenceBatch,
    train_fraction: float = 0.8,
) -> tuple[SequenceBatch, SequenceBatch]:
    """Split one sequence batch into chronological inner partitions."""
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not 0.0 < train_fraction < 1.0
    ):
        raise ValueError("train_fraction must be between zero and one")
    if not (
        len(batch.features) == len(batch.targets) == len(batch.keys)
        and batch.features.index.equals(batch.targets.index)
        and batch.features.index.equals(batch.keys.index)
    ):
        raise ValueError("sequence batch components must be aligned")
    boundary = int(len(batch.features) * train_fraction)
    if boundary == 0 or boundary == len(batch.features):
        raise ValueError("train_fraction must create two non-empty partitions")
    return _slice_batch(batch, slice(0, boundary)), _slice_batch(batch, slice(boundary, None))


def fit_return_scaler(
    dataset: pd.DataFrame,
    ticker: str,
    through_date: pd.Timestamp,
) -> StandardScaler:
    """Fit a return scaler on evaluation history through one origin date."""
    validate_model_dataset(dataset)
    validated_ticker = _validate_ticker(ticker)
    if not dataset["ticker"].eq(validated_ticker).any():
        raise ValueError(f"ticker {validated_ticker!r} is absent from dataset")

    cutoff = pd.Timestamp(through_date)
    history = dataset.loc[
        dataset["ticker"].eq(validated_ticker)
        & dataset["split"].isin(EVALUATION_SPLITS)
        & dataset["date"].le(cutoff),
        RETURN_COLUMN,
    ]
    if history.empty:
        raise ValueError("return history through cutoff must not be empty")
    require_finite_numeric(history, RETURN_COLUMN)
    return StandardScaler().fit(history.to_numpy(dtype="float64").reshape(-1, 1))


def _finite_numeric_values(values: pd.DataFrame | pd.Series, label: str) -> np.ndarray:
    """Return finite float values from one pandas object."""
    numeric = values.to_numpy()
    if numeric.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be numeric")
    result = numeric.astype("float64")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def transform_features(features: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Scale lag features and add the univariate channel dimension."""
    values = _finite_numeric_values(features, "features")
    transformed = cast(np.ndarray, scaler.transform(values.reshape(-1, 1)))
    return transformed.reshape(len(features), len(features.columns), 1).astype("float64")


def transform_targets(targets: pd.Series, scaler: StandardScaler) -> np.ndarray:
    """Scale target returns into one flat numeric vector."""
    values = _finite_numeric_values(targets, "targets")
    transformed = cast(np.ndarray, scaler.transform(values.reshape(-1, 1)))
    return transformed.reshape(-1).astype("float64")


def inverse_transform_targets(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Restore one target vector to its original return scale."""
    array = np.asarray(values)
    valid_shape = array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1)
    if not valid_shape or array.dtype.kind not in "iuf":
        raise ValueError("values must be a numeric vector")
    numeric = array.astype("float64")
    if not np.isfinite(numeric).all():
        raise ValueError("values must contain only finite values")
    restored = cast(np.ndarray, scaler.inverse_transform(numeric.reshape(-1, 1)))
    return restored.reshape(-1).astype("float64")
