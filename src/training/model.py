import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Train a model on the given data."""
    model = LogisticRegression()
    model.fit(X, y)
    return model


def evaluate_model(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X)
    return {"accuracy": accuracy_score(y, y_pred)}
