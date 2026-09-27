"""Tests for configurable univariate LSTM forecasting models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from data.sequences import lag_columns
from models.lstm import (
    LSTMConfig,
    LSTMEnsemble,
    LSTMModel,
    LSTMTrainer,
    build_lstm_grid,
)


@dataclass(frozen=True)
class FitRecord:
    """Capture one fake-network fit invocation"""

    features: np.ndarray
    targets: np.ndarray
    epochs: int
    batch_size: int
    shuffle: bool
    validation_data: tuple[np.ndarray, np.ndarray] | None
    callback_count: int
    verbose: int


class FakeHistory:
    """Expose controlled Keras-compatible history values."""

    def __init__(self, history: Mapping[str, list[float]]) -> None:
        self.history = history


class RecordingNetwork:
    """Record training options and return controlled predictions."""

    def __init__(
        self,
        *,
        validation_losses: Sequence[float] = (0.9, 0.4, 0.5),
        prediction: object | Callable[[np.ndarray], object] | None = None,
        parameter_count: int = 73,
    ) -> None:
        self.validation_losses = list(validation_losses)
        self.prediction = np.array([[0.0]]) if prediction is None else prediction
        self._parameter_count = parameter_count
        self.fit_records: list[FitRecord] = []
        self.predict_inputs: list[np.ndarray] = []

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
    ) -> FakeHistory:
        """Record one training call and return loss history"""
        self.fit_records.append(
            FitRecord(
                features=features.copy(),
                targets=targets.copy(),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=validation_data,
                callback_count=len(callbacks or ()),
                verbose=verbose,
            )
        )
        return FakeHistory({"val_loss": self.validation_losses})

    def predict(self, features: np.ndarray, *, verbose: int = 0) -> object:
        """Return a controlled prediction for supplied features"""
        assert verbose == 0
        self.predict_inputs.append(features.copy())
        if callable(self.prediction):
            return self.prediction(features)
        return self.prediction

    def count_params(self) -> int:
        """Return the controlled trainable parameter count"""
        return self._parameter_count


class RecordingBuilder:
    """Create fresh recording networks for each trainer build."""

    def __init__(self, factory: Callable[[], RecordingNetwork] | None = None) -> None:
        self.factory = factory or RecordingNetwork
        self.calls: list[tuple[LSTMConfig, int]] = []
        self.networks: list[RecordingNetwork] = []

    def __call__(self, config: LSTMConfig, seed: int) -> RecordingNetwork:
        """Build and retain one fresh controlled network"""
        self.calls.append((config, seed))
        network = self.factory()
        self.networks.append(network)
        return network


def identity_scaler() -> StandardScaler:
    """Return a scaler with exactly zero mean and unit scale"""
    return StandardScaler().fit(np.array([[-1.0], [1.0]]))


def training_arrays(window: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return finite univariate arrays for controlled training"""
    features = np.arange(16, dtype="float64").reshape(8, 2, 1)
    if window != 2:
        features = np.arange(8 * window, dtype="float64").reshape(8, window, 1)
    targets = np.linspace(-0.02, 0.02, num=8, dtype="float64")
    return features, targets


def constant_prediction(value: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return one constant backend forecast per observation"""

    def predict(features: np.ndarray) -> np.ndarray:
        return np.full((len(features), 1), value, dtype="float64")

    return predict


def fit_fake_model(
    *,
    config: LSTMConfig | None = None,
    prediction: object | Callable[[np.ndarray], object] | None = None,
    name: str = "lstm_seed",
) -> tuple[LSTMModel, RecordingNetwork]:
    """Fit one model backed by a controlled prediction network"""
    selected_config = config or LSTMConfig(window=2, units=4, dropout=0.0)
    builder = RecordingBuilder(lambda: RecordingNetwork(prediction=prediction))
    trainer = LSTMTrainer(network_builder=builder)
    features, targets = training_arrays(selected_config.window)
    model = trainer.fit(
        selected_config,
        seed=0,
        features=features,
        targets=targets,
        scaler=identity_scaler(),
        feature_columns=lag_columns(selected_config.window),
        epochs=2,
        name=name,
    )
    return model, builder.networks[0]


def test_build_lstm_grid_returns_all_fixed_configurations() -> None:
    """Return twelve deterministic window units dropout combinations"""
    expected = tuple(
        LSTMConfig(window=window, units=units, dropout=dropout)
        for window in (5, 21, 63)
        for units in (16, 32)
        for dropout in (0.0, 0.2)
    )

    assert build_lstm_grid() == expected
    assert len(set(build_lstm_grid())) == 12


@pytest.mark.parametrize("field", ["window", "units"])
@pytest.mark.parametrize("invalid_value", [True, 0, -1, 1.5, "5"])
def test_lstm_config_rejects_invalid_integer_fields(
    field: str,
    invalid_value: object,
) -> None:
    """Reject boolean noninteger and nonpositive dimensions"""
    values: dict[str, object] = {"window": 5, "units": 16, "dropout": 0.2}
    values[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        LSTMConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_dropout", [True, -0.1, 1.0, 1.1, "0.2", None])
def test_lstm_config_rejects_invalid_dropout(invalid_dropout: object) -> None:
    """Reject nonnumeric and out-of-range dropout values"""
    with pytest.raises(ValueError, match="dropout"):
        LSTMConfig(window=5, units=16, dropout=invalid_dropout)  # type: ignore[arg-type]


def test_trainer_selects_one_based_epoch_with_minimum_validation_loss() -> None:
    """Choose best chronological validation epoch from history"""
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    builder = RecordingBuilder()
    trainer = LSTMTrainer(network_builder=builder)
    features, targets = training_arrays()

    best_epoch = trainer.determine_best_epoch(
        config,
        seed=3,
        train_features=features[:6],
        train_targets=targets[:6],
        validation_features=features[6:],
        validation_targets=targets[6:],
    )

    assert best_epoch == 2
    assert builder.calls == [(config, 3)]
    record = builder.networks[0].fit_records[0]
    assert record.epochs == 200
    assert record.batch_size == 32
    assert record.shuffle is False
    assert record.verbose == 0
    assert record.callback_count == 1
    assert record.features.tolist() == features[:6].tolist()
    assert record.targets.tolist() == targets[:6].tolist()
    assert record.validation_data is not None
    assert record.validation_data[0].tolist() == features[6:].tolist()
    assert record.validation_data[1].tolist() == targets[6:].tolist()


def test_trainer_rebuilds_model_for_full_training() -> None:
    """Build fresh same-seed network for fixed full epochs"""
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    builder = RecordingBuilder()
    trainer = LSTMTrainer(network_builder=builder)
    features, targets = training_arrays()

    trainer.determine_best_epoch(
        config,
        seed=4,
        train_features=features[:6],
        train_targets=targets[:6],
        validation_features=features[6:],
        validation_targets=targets[6:],
    )
    model = trainer.fit(
        config,
        seed=4,
        features=features,
        targets=targets,
        scaler=identity_scaler(),
        feature_columns=lag_columns(2),
        epochs=2,
        name="lstm_AAA_seed_04",
    )

    assert len(builder.networks) == 2
    assert builder.calls == [(config, 4), (config, 4)]
    record = builder.networks[1].fit_records[0]
    assert record.epochs == 2
    assert record.batch_size == 32
    assert record.shuffle is False
    assert record.validation_data is None
    assert record.callback_count == 0
    assert model.name == "lstm_AAA_seed_04"
    assert model.config == config
    assert model.feature_columns == ("lag_1", "lag_0")
    assert model.parameter_count == 73


@pytest.mark.parametrize("seed", [True, -1, 10, 1.5, "1"])
def test_trainer_rejects_invalid_seeds(seed: object) -> None:
    """Restrict trainer seeds to exact experiment range"""
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()
    trainer = LSTMTrainer(network_builder=RecordingBuilder())

    with pytest.raises(ValueError, match="seed"):
        trainer.fit(
            config,
            seed=seed,  # type: ignore[arg-type]
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=lag_columns(2),
            epochs=2,
            name="lstm",
        )


@pytest.mark.parametrize("epochs", [True, 0, -1, 201, 1.5, "2"])
def test_trainer_rejects_invalid_epoch_counts(epochs: object) -> None:
    """Restrict full training to configured epoch bounds"""
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()
    trainer = LSTMTrainer(network_builder=RecordingBuilder())

    with pytest.raises(ValueError, match="epochs"):
        trainer.fit(
            config,
            seed=0,
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=lag_columns(2),
            epochs=epochs,  # type: ignore[arg-type]
            name="lstm",
        )


@pytest.mark.parametrize(
    ("feature_shape", "target_length", "message"),
    [
        ((8, 2), 8, "three-dimensional"),
        ((8, 2, 2), 8, "univariate"),
        ((8, 3, 1), 8, "window"),
        ((8, 2, 1), (8, 1), "one-dimensional"),
        ((8, 2, 1), 7, "length"),
        ((0, 2, 1), 0, "empty"),
    ],
)
def test_trainer_rejects_invalid_training_shapes(
    feature_shape: tuple[int, ...],
    target_length: int | tuple[int, ...],
    message: str,
) -> None:
    """Reject arrays incompatible with one configured sequence"""
    trainer = LSTMTrainer(network_builder=RecordingBuilder())
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features = np.zeros(feature_shape, dtype="float64")
    targets = np.zeros(target_length, dtype="float64")

    with pytest.raises(ValueError, match=message):
        trainer.fit(
            config,
            seed=0,
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=lag_columns(2),
            epochs=2,
            name="lstm",
        )


@pytest.mark.parametrize("array_name", ["features", "targets"])
def test_trainer_rejects_nonnumeric_training_arrays(array_name: str) -> None:
    """Reject object arrays before TensorFlow training"""
    trainer = LSTMTrainer(network_builder=RecordingBuilder())
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()
    if array_name == "features":
        features = features.astype(object)
        features[0, 0, 0] = "invalid"
    else:
        targets = targets.astype(object)
        targets[0] = "invalid"

    with pytest.raises(ValueError, match="numeric"):
        trainer.fit(
            config,
            seed=0,
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=lag_columns(2),
            epochs=2,
            name="lstm",
        )


@pytest.mark.parametrize("array_name", ["features", "targets"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_trainer_rejects_nonfinite_training_arrays(
    array_name: str,
    invalid_value: float,
) -> None:
    """Reject nonfinite values before TensorFlow training"""
    trainer = LSTMTrainer(network_builder=RecordingBuilder())
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()
    if array_name == "features":
        features[0, 0, 0] = invalid_value
    else:
        targets[0] = invalid_value

    with pytest.raises(ValueError, match=array_name):
        trainer.fit(
            config,
            seed=0,
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=lag_columns(2),
            epochs=2,
            name="lstm",
        )


@pytest.mark.parametrize(
    "validation_losses",
    [(), (0.5, np.nan), (0.5, np.inf)],
)
def test_trainer_rejects_invalid_validation_history(
    validation_losses: tuple[float, ...],
) -> None:
    """Reject empty and nonfinite early stopping history"""
    builder = RecordingBuilder(lambda: RecordingNetwork(validation_losses=validation_losses))
    trainer = LSTMTrainer(network_builder=builder)
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()

    with pytest.raises(ValueError, match="val_loss"):
        trainer.determine_best_epoch(
            config,
            seed=0,
            train_features=features[:6],
            train_targets=targets[:6],
            validation_features=features[6:],
            validation_targets=targets[6:],
        )


def test_trainer_requires_standard_lag_columns() -> None:
    """Reject fitted models with ambiguous feature ordering"""
    trainer = LSTMTrainer(network_builder=RecordingBuilder())
    config = LSTMConfig(window=2, units=4, dropout=0.0)
    features, targets = training_arrays()

    with pytest.raises(ValueError, match="feature_columns"):
        trainer.fit(
            config,
            seed=0,
            features=features,
            targets=targets,
            scaler=identity_scaler(),
            feature_columns=("lag_0", "lag_1"),
            epochs=2,
            name="lstm",
        )


def test_lstm_model_requires_standard_lag_columns() -> None:
    """Protect model construction from ambiguous lag ordering"""
    with pytest.raises(ValueError, match="feature_columns"):
        LSTMModel(
            RecordingNetwork(),
            identity_scaler(),
            LSTMConfig(window=2, units=4, dropout=0.0),
            ("lag_0", "lag_1"),
            name="lstm",
        )


def test_lstm_model_predicts_current_lag_on_original_scale() -> None:
    """Scale reshape forecast and invert while preserving index"""

    def echo_current(features: np.ndarray) -> np.ndarray:
        return features[:, -1, 0].reshape(-1, 1)

    model, network = fit_fake_model(prediction=echo_current, name="lstm_AAA_seed_00")
    observations = pd.DataFrame(
        {
            "lag_1": [-0.02, 0.01],
            "lag_0": [0.03, -0.04],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        },
        index=pd.Index([20, 40]),
    )
    original = observations.copy(deep=True)

    result = model.predict(observations)

    assert result.name == "predicted_return"
    assert result.index.tolist() == [20, 40]
    assert result.tolist() == pytest.approx([0.03, -0.04])
    assert network.predict_inputs[0].shape == (2, 2, 1)
    pd.testing.assert_frame_equal(observations, original)


def test_lstm_model_returns_empty_aligned_predictions() -> None:
    """Return empty named prediction without backend invocation"""
    model, network = fit_fake_model()
    observations = pd.DataFrame(columns=lag_columns(2), index=pd.Index([], dtype="int64"))

    result = model.predict(observations)

    assert result.empty
    assert result.dtype == "float64"
    assert result.name == "predicted_return"
    assert result.index.equals(observations.index)
    assert network.predict_inputs == []


def test_lstm_model_rejects_missing_lag_columns() -> None:
    """Reject observations without complete configured window"""
    model, _ = fit_fake_model()

    with pytest.raises(ValueError, match="missing columns.*lag_1"):
        model.predict(pd.DataFrame({"lag_0": [0.01]}))


@pytest.mark.parametrize("invalid_value", ["invalid", np.nan, np.inf, -np.inf])
def test_lstm_model_rejects_invalid_feature_values(invalid_value: object) -> None:
    """Reject nonnumeric and nonfinite lag values"""
    model, _ = fit_fake_model()
    observations = pd.DataFrame({"lag_1": [0.01], "lag_0": [0.02]})
    if isinstance(invalid_value, str):
        observations["lag_0"] = observations["lag_0"].astype(object)
    observations.loc[0, "lag_0"] = invalid_value  # type: ignore[call-overload]

    with pytest.raises(ValueError, match="features"):
        model.predict(observations)


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (np.array([[0.0], [0.1]]), "length"),
        (np.array([["invalid"]]), "numeric"),
        (np.array([[np.nan]]), "finite"),
        (np.array([[np.inf]]), "finite"),
        (np.array([[[0.0]]]), "vector"),
    ],
)
def test_lstm_model_rejects_invalid_backend_predictions(
    prediction: np.ndarray,
    message: str,
) -> None:
    """Reject malformed backend prediction values and shapes"""
    model, _ = fit_fake_model(prediction=prediction)
    observations = pd.DataFrame({"lag_1": [0.01], "lag_0": [0.02]})

    with pytest.raises(ValueError, match=message):
        model.predict(observations)


@pytest.mark.parametrize("name", ["", "   ", 1, None])
def test_lstm_model_rejects_invalid_names(name: object) -> None:
    """Reject empty and nonstring model identifiers"""
    with pytest.raises(ValueError, match="model name"):
        fit_fake_model(name=cast(str, name))


def test_lstm_ensemble_averages_ten_aligned_models() -> None:
    """Average ten seed forecasts into one stable prediction"""
    models = [
        fit_fake_model(prediction=constant_prediction(seed / 1_000), name=f"seed_{seed}")[0]
        for seed in range(10)
    ]
    ensemble = LSTMEnsemble(models)
    observations = pd.DataFrame(
        {"lag_1": [0.01, 0.02], "lag_0": [0.03, 0.04]},
        index=pd.Index([7, 9]),
    )

    result = ensemble.predict(observations)

    assert ensemble.name == "lstm"
    assert result.name == "predicted_return"
    assert result.index.tolist() == [7, 9]
    assert result.tolist() == pytest.approx([0.0045, 0.0045])


def test_lstm_ensemble_accepts_one_model() -> None:
    """Support the minimum nonempty ensemble boundary"""
    model, _ = fit_fake_model(prediction=constant_prediction(0.025))
    observations = pd.DataFrame({"lag_1": [0.01], "lag_0": [0.02]})

    result = LSTMEnsemble([model], name="single_seed").predict(observations)

    assert result.tolist() == pytest.approx([0.025])
    assert result.name == "predicted_return"


def test_lstm_ensemble_rejects_empty_models() -> None:
    """Reject an ensemble without forecasting members"""
    with pytest.raises(ValueError, match="at least one"):
        LSTMEnsemble([])


def test_lstm_ensemble_rejects_duplicate_model_instances() -> None:
    """Reject duplicate references masquerading as seed models"""
    model, _ = fit_fake_model()

    with pytest.raises(ValueError, match="distinct"):
        LSTMEnsemble([model, model])


def test_lstm_ensemble_rejects_inconsistent_configurations() -> None:
    """Reject members trained with different configurations"""
    first, _ = fit_fake_model(config=LSTMConfig(2, 4, 0.0), name="first")
    second, _ = fit_fake_model(config=LSTMConfig(3, 4, 0.0), name="second")

    with pytest.raises(ValueError, match="configuration"):
        LSTMEnsemble([first, second])


def test_lstm_ensemble_propagates_member_prediction_errors() -> None:
    """Reject ensemble when one member violates output contract"""
    valid, _ = fit_fake_model(prediction=np.array([[0.01]]), name="valid")
    invalid, _ = fit_fake_model(prediction=np.array([[np.nan]]), name="invalid")
    ensemble = LSTMEnsemble([valid, invalid])
    observations = pd.DataFrame({"lag_1": [0.01], "lag_0": [0.02]})

    with pytest.raises(ValueError, match="finite"):
        ensemble.predict(observations)


def test_tensorflow_lstm_trainer_smoke() -> None:
    """Train real Keras model and return one finite forecast"""
    config = LSTMConfig(window=2, units=2, dropout=0.0)
    trainer = LSTMTrainer()
    scaler = identity_scaler()
    raw_features = np.array(
        [
            [-0.04, -0.03],
            [-0.03, -0.02],
            [-0.02, -0.01],
            [-0.01, 0.00],
            [0.00, 0.01],
            [0.01, 0.02],
        ],
        dtype="float64",
    )
    features = scaler.transform(raw_features.reshape(-1, 1)).reshape(6, 2, 1)
    targets = scaler.transform(np.array([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03]).reshape(-1, 1))
    model = trainer.fit(
        config,
        seed=0,
        features=features,
        targets=targets.reshape(-1),
        scaler=scaler,
        feature_columns=lag_columns(2),
        epochs=2,
        name="lstm_smoke",
    )
    observations = pd.DataFrame(
        {"lag_1": [0.02], "lag_0": [0.03]},
        index=pd.Index([99]),
    )

    prediction = model.predict(observations)

    assert prediction.index.tolist() == [99]
    assert prediction.name == "predicted_return"
    assert np.isfinite(prediction.iloc[0])
