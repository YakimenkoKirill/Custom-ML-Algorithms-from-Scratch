# Custom-ML-Algorithms-from-Scratch: MiniBatchSGDRegressor

## Production-ready, fully vectorized implementation of Machine Learning algorithms built from scratch using NumPy.

## Features
- 100% test coverage(pytest)
- Static type checking(mypy >= 1.9.0) & code formatting(ruff >= 0.3.0)
- Strict OOP interfaces adhering to Scikit-Learn conventions

## Content
- About the project
- Mathematical description
- Architecture and structure of the package
- Perfomance benchmarks
- Installation and launch
- Example of use

## About the project
MiniBatchSGDRegressor is a modular and scalable implementation of linear regression with mini-batch gradient descent and L-2 regularization.
The project was developed without using third-party ML frameworks for training(Numpy) and strictly complies with the Scikit-Learn API(fit, predict)

## Mathematical description

#### Target function(MSE + L2)
$$L(w, b) = \frac{1}{2N} \sum_{i=1}^N \left( x_i w + b - y_i \right)^2 + \frac{\lambda}{2} \Vert{}w\Vert{}_2^2$$

#### Vectorized Gradients
$$\nabla_w L = \frac{2}{B} X_{\text{batch}}^T \left( X_{\text{batch}} w + b - y_{\text{batch}} \right) + \lambda w$$

$$\nabla_b L = \frac{2}{B} \sum_{i=1}^{B} e_i$$

#### Parametr Update Rule
$$w \leftarrow w - \eta \cdot \nabla_w L$$

$$b \leftarrow b - \eta \cdot \nabla_b L$$

где $\eta$ — темп обучения (`learning_rate`), $\lambda$ — коэффициент регуляризации (`l2_param`).

## Architecture and structure of the package
```bash
├── .github/
│   └── workflows/
│       └── ci.yml
├── custom_ml/
│   ├── base.py
│   ├── linear_model/
│   │   └── sgd_regressor.py
│   └── utils/
│       └── metrics.py
├── tests/
│   ├── test_base.py
│   ├── test_sgd_regressor.py
│   └── test_metrics.py
├── benchmarks/
│   ├── benchmark_vs_sklearn.py
│   └── results/
├── pyproject.toml
├── .gitignore
└── README.md
```

## Perfomance benchmarks
```markdown
| Metrics | Custom_SGDRegressor | Sklearn_SGDRegressor |
| Training time (s) | 0.7508 | 0.2205 |
| Memory peak (MB) | 1.88 | 1.07 |
| MSE | 1.2424 | 5.9107 |
| MAE | 0.8894 | 1.9408 |
| R2 | 1.0000 | 0.9999 |
```

## Installation and launch

#### 1.Cloning the repository and setting up the environment:
```bash
git clone [git@github.com:YakimenkoKirill/Custom-ML-Algorithms-from-Scratch.git](https://github.com/YakimenkoKirill/Custom-ML-Algorithms-from-Scratch.git)
cd Custom-ML-Algorithms-from-Scratch
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

#### 2.Running static analysis and tests:
```bash
ruff check .
mypy .
pytest -v
```

#### 3.Launching performance benchmarking:
```bash
python3 benchmarks/benchmark_vs_sklearn.py
```

## Example of use
```bash
import numpy as np
from custom_ml.linear_model.sgd_regressor import MiniBatchSGDRegressor
from custom_ml.utils.metrics import r2_score

X = np.random.randn(1000, 10)
true_weights = np.random.randn(10)
y = X @ true_weights + 0.1 * np.random.randn(1000)

model = MiniBatchSGDRegressor(
    learning_rate=0.01,
    n_epochs=50,
    batch_size=32,
    l2_param=0.01,
    random_state=42
)
model.fit(X, y)

predictions = model.predict(X)
print(f"R2 Score: {r2_score(y, predictions):.4f}")
```