from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np


class BaseEstimator(ABC):
    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in self.__dict__:
            if not key.endswith("_"):
                params[key] = getattr(self, key)
        return params

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self}")
        return self


class BaseRegressor(BaseEstimator):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from custom_ml.utils.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
