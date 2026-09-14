import numpy as np
import pytest
from sklearn.metrics import (
    mean_absolute_error as sklearn_mae,
)
from sklearn.metrics import (
    mean_squared_error as sklearn_mse,
)
from sklearn.metrics import (
    r2_score as sklearn_r2,
)

from custom_ml.utils.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def test_metrics_match_sklearn() -> None:
    y_true = np.array([1.5, -0.5, 3.0, 4.2])
    y_pred = np.array([1.4, -0.2, 2.8, 4.0])

    assert np.isclose(mean_squared_error(y_true, y_pred), sklearn_mse(y_true, y_pred))
    assert np.isclose(mean_absolute_error(y_true, y_pred), sklearn_mae(y_true, y_pred))
    assert np.isclose(r2_score(y_true, y_pred), sklearn_r2(y_true, y_pred))


def test_metrics_shape_mismatch() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred)
