"""Configurable univariate LSTM forecasting models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from data.sequences import (
    inverse_transform_targets,
    lag_columns,
    transform_features,
)
from evaluation.contracts import PREDICTION_NAME
from models.interfaces import validate_model_name

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
MIN_SEED = 0
MAX_SEED = 9


class _TrainingHistory(Protocol):
    """Minimal history surface returned by Keras training."""

    @property
    def history(self) -> Mapping[str, object]:
        """Return recorded metric values."""
        ...


class _TrainableNetwork(Protocol):
    """Minimal neural-network surface required by the trainer."""

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        callbacks: Sequence[object] | None = None,
        verbose: int = 0,
    ) -> _TrainingHistory:
        """Train the network and return metric history."""
        ...

    def predict(self, features: np.ndarray, *, verbose: int = 0) -> object:
        """Return scaled forecasts for supplied sequences."""
        ...

    def count_params(self) -> int:
        """Return the number of trainable and fixed parameters."""
        ...


NetworkBuilder = Callable[["LSTMConfig", int], _TrainableNetwork]


@dataclass(frozen=True)
class LSTMConfig:
    """Describe one univariate LSTM candidate configuration."""

    window: int
    units: int
    dropout: float

    def __post_init__(self) -> None:
        if type(self.window) is not int or self.window <= 0:
            raise ValueError("window must be a positive integer")
        if type(self.units) is not int or self.units <= 0:
            raise ValueError("units must be a positive integer")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("dropout must be a numeric value")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")
        object.__setattr__(self, "dropout", float(self.dropout))


def build_lstm_grid() -> tuple[LSTMConfig, ...]:
    """Return the fixed univariate LSTM candidate grid."""
    return tuple(
        LSTMConfig(window=window, units=units, dropout=dropout)
        for window in (5, 21, 63)
        for units in (16, 32)
        for dropout in (0.0, 0.2)
    )


def _validate_seed(seed: object) -> int:
    """Return an integer seed from the fixed experiment range."""
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from {MIN_SEED} through {MAX_SEED}")
    return seed


def _validate_training_arrays(
    config: LSTMConfig,
    features: np.ndarray,
    targets: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite arrays matching one univariate configuration."""
    feature_values = np.asarray(features)
    target_values = np.asarray(targets)
    if feature_values.ndim != 3:
        raise ValueError(f"{label} features must be three-dimensional")
    if feature_values.shape[2] != 1:
        raise ValueError(f"{label} features must have one univariate channel")
    if feature_values.shape[1] != config.window:
        raise ValueError(f"{label} feature window must match configuration")
    if target_values.ndim != 1:
        raise ValueError(f"{label} targets must be one-dimensional")
    if len(feature_values) != len(target_values):
        raise ValueError(f"{label} feature and target length must match")
    if len(feature_values) == 0:
        raise ValueError(f"{label} arrays must not be empty")
    if feature_values.dtype.kind not in "iuf" or target_values.dtype.kind not in "iuf":
        raise ValueError(f"{label} arrays must be numeric")
    numeric_features = feature_values.astype("float64")
    numeric_targets = target_values.astype("float64")
    if not np.isfinite(numeric_features).all():
        raise ValueError(f"{label} features must contain only finite values")
    if not np.isfinite(numeric_targets).all():
        raise ValueError(f"{label} targets must contain only finite values")
    return numeric_features, numeric_targets


def _build_keras_network(config: LSTMConfig, seed: int) -> _TrainableNetwork:
    """Build one deterministic compiled Keras regression network."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    network = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.window, 1)),
            tf.keras.layers.LSTM(config.units, dropout=config.dropout),
            tf.keras.layers.Dense(1),
        ]
    )
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return cast(_TrainableNetwork, network)


class LSTMModel:
    """Adapt one fitted univariate network to ForecastModel."""

    def __init__(
        self,
        network: _TrainableNetwork,
        scaler: StandardScaler,
        config: LSTMConfig,
        feature_columns: Sequence[str],
        *,
        name: str,
    ) -> None:
        columns = tuple(feature_columns)
        if columns != lag_columns(config.window):
            raise ValueError("feature_columns must match configured lag order")
        self._network = network
        self._scaler = scaler
        self._config = config
        self._feature_columns = columns
        self._name = validate_model_name(name)
        self._parameter_count = int(network.count_params())

    @property
    def name(self) -> str:
        """Return the stable model identifier."""
        return self._name

    @property
    def config(self) -> LSTMConfig:
        """Return the fitted candidate configuration."""
        return self._config

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return lag columns expected by prediction inputs."""
        return self._feature_columns

    @property
    def parameter_count(self) -> int:
        """Return the fitted network parameter count."""
        return self._parameter_count

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Predict next returns from raw lag observations."""
        missing = set(self._feature_columns) - set(observations.columns)
        if missing:
            raise ValueError(f"observations missing columns: {sorted(missing)}")
        if observations.empty:
            return pd.Series(
                index=observations.index,
                dtype="float64",
                name=PREDICTION_NAME,
            )

        features = observations.loc[:, self._feature_columns].copy()
        transformed = transform_features(features, self._scaler)
        predicted = np.asarray(self._network.predict(transformed, verbose=0))
        if predicted.ndim == 0 or len(predicted) != len(observations):
            raise ValueError("prediction length must match observations")
        restored = inverse_transform_targets(predicted, self._scaler)
        return pd.Series(
            restored,
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class LSTMEnsemble:
    """Average aligned forecasts from one LSTM seed ensemble."""

    def __init__(self, models: Sequence[LSTMModel], name: str = "lstm") -> None:
        members = tuple(models)
        if not members:
            raise ValueError("ensemble must contain at least one model")
        if len({id(model) for model in members}) != len(members):
            raise ValueError("ensemble models must be distinct instances")
        first = members[0]
        if any(model.config != first.config for model in members[1:]):
            raise ValueError("ensemble models must share one configuration")
        self._models = members
        self._name = validate_model_name(name)

    @property
    def name(self) -> str:
        """Return the stable ensemble identifier."""
        return self._name

    def predict(self, observations: pd.DataFrame) -> pd.Series:
        """Return the arithmetic mean of aligned member predictions."""
        member_predictions = [model.predict(observations) for model in self._models]
        values = np.column_stack(
            [prediction.to_numpy(dtype="float64") for prediction in member_predictions]
        )
        return pd.Series(
            values.mean(axis=1),
            index=observations.index,
            dtype="float64",
            name=PREDICTION_NAME,
        )


class LSTMTrainer:
    """Train deterministic LSTM candidates with an injectable backend."""

    def __init__(self, network_builder: NetworkBuilder | None = None) -> None:
        self._network_builder = network_builder or _build_keras_network

    def determine_best_epoch(
        self,
        config: LSTMConfig,
        seed: int,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
    ) -> int:
        """Return the one-based epoch with minimum inner validation loss."""
        validated_seed = _validate_seed(seed)
        train_x, train_y = _validate_training_arrays(
            config,
            train_features,
            train_targets,
            "train",
        )
        validation_x, validation_y = _validate_training_arrays(
            config,
            validation_features,
            validation_targets,
            "validation",
        )
        network = self._network_builder(config, validated_seed)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=0.0,
            restore_best_weights=True,
        )
        history = network.fit(
            train_x,
            train_y,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_data=(validation_x, validation_y),
            callbacks=[early_stopping],
            verbose=0,
        )
        losses = np.asarray(history.history.get("val_loss", ()))
        if losses.ndim != 1 or len(losses) == 0 or losses.dtype.kind not in "iuf":
            raise ValueError("val_loss history must be a non-empty numeric vector")
        numeric_losses = losses.astype("float64")
        if not np.isfinite(numeric_losses).all():
            raise ValueError("val_loss history must contain only finite values")
        return int(np.argmin(numeric_losses)) + 1

    def fit(
        self,
        config: LSTMConfig,
        seed: int,
        features: np.ndarray,
        targets: np.ndarray,
        scaler: StandardScaler,
        feature_columns: Sequence[str],
        epochs: int,
        name: str,
    ) -> LSTMModel:
        """Fit one candidate on all supplied scaled sequences."""
        validated_seed = _validate_seed(seed)
        if type(epochs) is not int or not 1 <= epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be an integer from 1 through {MAX_EPOCHS}")
        columns = tuple(feature_columns)
        if columns != lag_columns(config.window):
            raise ValueError("feature_columns must match configured lag order")
        validated_name = validate_model_name(name)
        train_x, train_y = _validate_training_arrays(config, features, targets, "train")
        network = self._network_builder(config, validated_seed)
        network.fit(
            train_x,
            train_y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0,
        )
        return LSTMModel(
            network,
            scaler,
            config,
            columns,
            name=validated_name,
        )
