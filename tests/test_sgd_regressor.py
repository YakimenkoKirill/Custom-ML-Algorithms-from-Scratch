import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from custom_ml.linear_model.sgd_regressor import MiniBatchSGDRegressor


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def test_unfitted_model_raises_error() -> None:
    model = MiniBatchSGDRegressor()
    X = np.random.randn(10, 5)
    with pytest.raises(AttributeError):
        model.predict(X)


def test_loss_history_decreases(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_data
    model = MiniBatchSGDRegressor(n_epochs=50, learning_rate=0.01, random_state=42)
    model.fit(X, y)

    assert len(model.loss_history_) > 0
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_l2_regularization_shrinks_weights(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_data

    model_no_reg = MiniBatchSGDRegressor(
        l2_param=0.0, n_epochs=50, random_state=42
    ).fit(X, y)
    model_reg = MiniBatchSGDRegressor(
        l2_param=1.0, n_epochs=50, random_state=42
    ).fit(X, y)

    assert model_no_reg.weights_ is not None and model_reg.weights_ is not None
    norm_no_reg = np.linalg.norm(model_no_reg.weights_)
    norm_reg = np.linalg.norm(model_reg.weights_)

    assert norm_reg < norm_no_reg


def test_sklearn_benchmark_comparison(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_data

    custom_model = MiniBatchSGDRegressor(
        learning_rate=0.01,
        n_epochs=100,
        batch_size=32,
        l2_param=0.01,
        random_state=42,
    ).fit(X, y)

    sklearn_model = SklearnSGDRegressor(
        learning_rate="constant",
        eta0=0.01,
        max_iter=100,
        alpha=0.01,
        random_state=42,
    ).fit(X, y)

    custom_r2 = r2_score(y, custom_model.predict(X))
    sklearn_r2 = r2_score(y, sklearn_model.predict(X))

    assert custom_r2 > 0.85
    assert np.isclose(custom_r2, sklearn_r2, atol=0.1)