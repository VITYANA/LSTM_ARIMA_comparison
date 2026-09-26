"""Shared contracts for forecast evaluation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS_FACTOR = 10_000.0
EVALUATION_SPLITS = frozenset({"train", "validation"})
PREDICTION_COLUMNS = (
    "date",
    "target_date",
    "ticker",
    "split",
    "model",
    "actual_return",
    "predicted_return",
)


def require_finite_numeric(series: pd.Series, label: str) -> None:
    """Require a numeric series containing only finite values."""
    if series.dtype.kind not in "iufc":
        raise ValueError(f"{label} must be numeric")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{label} must contain only finite values")
