import time
import tracemalloc
from typing import Any
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from custom_ml.linear_model.sgd_regressor import MiniBatchSGDRegressor
from sklearn.preprocessing import StandardScaler
from custom_ml.utils.metrics import mean_absolute_error, mean_squared_error, r2_score


def benchmark_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    tracemalloc.start()
    start_time = time.perf_counter()

    model.fit(X_train ,y_train)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    memory_mb = peak_memory / (1024 * 1024)

    return {
        "time_sec": elapsed_time, 
        "memory_mb": memory_mb,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

def main() -> None:
    print("dataset generation 100 000 * 50")
    X, y = make_regression(
        n_samples = 100000, n_features= 50, noise = 0.1, random_state= 21
    )
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )


    print("Custom MinibatchSGDRegressor")
    custom_model = MiniBatchSGDRegressor(
        learning_rate=0.01,
        n_epochs = 20,
        batch_size= 256, 
        l2_param=0.01,
        random_state=21
    )

    custom_result = benchmark_model(
        custom_model, X_train, X_test, y_train, y_test
    )

    print("Sklearn SGDRegressor")
    sklearn_model = SklearnSGDRegressor(
        learning_rate="constant",
        eta0=0.01,
        max_iter=20,
        alpha=0.01,
        random_state=21,
        tol = 1e-4
    )

    sklearn_result = benchmark_model(
        sklearn_model, X_train, X_test, y_train, y_test
        )

    results_dir = Path("benchmarks/results")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_path = results_dir / f"report_{timestamp}.txt"

    report_content = (
        f"Results ({timestamp})\n"
        + "=" * 65 + "\n"
        + f"{'Metrics':<20} | {'Custom_SGDRegressor':<20} | {'Sklearn_SGDRegressor':<20}\n"
        + "-" * 65 + "\n"
        + f"{'Training time (s)':<20} | {custom_result['time_sec']:<20.4f} | {sklearn_result['time_sec']:<20.4f}\n"
        + f"{'Memory peak (MB)':<20} | {custom_result['memory_mb']:<20.2f} | {sklearn_result['memory_mb']:<20.2f}\n"
        + f"{'MSE':<20} | {custom_result['mse']:<20.4f} | {sklearn_result['mse']:<20.4f}\n"
        + f"{'MAE':<20} | {custom_result['mae']:<20.4f} | {sklearn_result['mae']:<20.4f}\n"
        + f"{'R2':<20} | {custom_result['r2']:<20.4f} | {sklearn_result['r2']:<20.4f}\n"
        + "=" * 65 + "\n"
    )


    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report_content)

    print(report_content)


if __name__ == "__main__":
    main()