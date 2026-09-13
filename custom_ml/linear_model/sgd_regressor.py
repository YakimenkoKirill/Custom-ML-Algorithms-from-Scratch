from typing import Self
import numpy as np

from custom_ml.base import BaseRegressor
from custom_ml.utils.metrics import mean_squared_error


class MiniBatchSGDRegressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float | str = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        l2_param: float = 0.0,
        random_state: int = 21,
        tol: float = 1e-4,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_param = l2_param
        self.random_state = random_state
        self.tol = tol
        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.loss_history_: list[float] = []

    def _init_weights(self, n_features: int) -> np.random.Generator:
        random_rn = np.random.default_rng(self.random_state)
        self.weights_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0
        return random_rn

    def _get_learning_rate(self, epoch: int) -> float:
        if isinstance(self.learning_rate, float):
            return self.learning_rate
        if self.learning_rate == "decay":
            return 0.1 / (1 + epoch * 0.1)
        raise ValueError(f"Unsupported learning_rate: {self.learning_rate}")

    def _compute_gradients(
        self, X_batch: np.ndarray, y_batch: np.ndarray
    ) -> tuple[np.ndarray, float]:
        assert self.weights_ is not None and self.bias_ is not None

        y_pred_batch = X_batch @ self.weights_ + self.bias_
        e = y_pred_batch - y_batch

        grad_w = 2 * (X_batch.T @ e) / X_batch.shape[0] + self.l2_param * self.weights_
        grad_b = float(2 * np.mean(e))

        return grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        random_rn = self._init_weights(n_features)
        self.loss_history_ = []

        assert self.weights_ is not None and self.bias_ is not None

        for i in range(self.n_epochs):
            eta = self._get_learning_rate(i)
            indices = random_rn.permutation(n_samples)

            for str_idx in range(0, n_samples, self.batch_size):
                end_idx = min(str_idx + self.batch_size, n_samples)
                batch_indices = indices[str_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad_w, grad_b = self._compute_gradients(X_batch, y_batch)
                self.weights_ = self.weights_ - eta * grad_w
                self.bias_ = self.bias_ - eta * grad_b

            current_y_pred = self.predict(X)
            current_loss = mean_squared_error(y, current_y_pred)
            self.loss_history_.append(current_loss)

            if i > 0:
                loss_diff = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if loss_diff < self.tol:
                    break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.weights_ is None or self.bias_ is None:
            raise AttributeError(
                "This model instance is not fitted yet. "
            )

        res: np.ndarray = (X @ self.weights_ + self.bias_).ravel()
        return res