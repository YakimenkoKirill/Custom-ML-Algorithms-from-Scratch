from typing import Self
import numpy as np
import pytest

from custom_ml.base import BaseRegressor


class DummyRegressor(BaseRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)


def test_cannot_instantiate_base_regressor() -> None:
    with pytest.raises(TypeError):
        BaseRegressor()  # type: ignore[abstract]


def test_dummy_regressor_contract() -> None:
    model = DummyRegressor()
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)

    fitted_model = model.fit(X, y)
    assert fitted_model is model

    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.array_equal(predictions, np.zeros(2, dtype=np.float64))