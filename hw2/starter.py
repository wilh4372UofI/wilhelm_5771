"""HW2 — Regression & Classification From Scratch: starter code.

Implement every function below by replacing ``raise NotImplementedError``.
Do not change any function signatures.

Conventions: ``X`` is (n, d), ``y`` is (n,), parameters are (d,); losses
and gradients are MEANS over the n samples.

Run the tests from the hw2/ directory:
    pytest tests/ -v
"""
# Chris Wilhelm - CS5722 - Homework 2 - 2026-10-04


import numpy as np  # noqa: F401  (provided for student solutions)
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.model_selection import cross_val_score  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401


# ---------------------------------------------------------------------------
# Part 1 — Linear regression from scratch
# ---------------------------------------------------------------------------
def mse_loss(X, y, theta):
    """Mean squared error of a linear model.

    loss = (1/n) * sum_i (X[i] @ theta - y[i])**2

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        y (np.ndarray): Targets, shape (n,).
        theta (np.ndarray): Parameters, shape (d,).

    Returns:
        float: The mean squared error (a Python float or 0-d array).
    """
    
    loss = (1/n) * sum_i (X[i] @ theta - y[i])**2
    return loss



def mse_gradient(X, y, theta):
    """Gradient of ``mse_loss`` with respect to ``theta``.

    grad = (2/n) * X.T @ (X @ theta - y)

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        y (np.ndarray): Targets, shape (n,).
        theta (np.ndarray): Parameters, shape (d,).

    Returns:
        np.ndarray: Gradient vector, shape (d,).
    """
    raise NotImplementedError


def gradient_descent(X, y, lr, epochs):
    """Batch gradient descent on ``mse_loss`` starting from theta = zeros.

    For each of ``epochs`` iterations:
      1. theta = theta - lr * mse_gradient(X, y, theta)
      2. append mse_loss(X, y, theta) to the loss history

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        y (np.ndarray): Targets, shape (n,).
        lr (float): Learning rate.
        epochs (int): Number of full-batch updates.

    Returns:
        tuple[np.ndarray, list[float]]:
            - theta: final parameters, shape (d,).
            - loss_history: list of length ``epochs`` with the loss after
              each update. On well-conditioned data with a sensible ``lr``
              this must be non-increasing (the tests check this).
    """
    raise NotImplementedError


def ridge_closed_form(X, y, alpha):
    """Ridge regression via the closed-form (normal-equation) solution.

    theta = (X.T @ X + alpha * I)^-1 @ X.T @ y

    where I is the (d, d) identity. With ``alpha = 0`` this reduces to
    ordinary least squares. This task uses the sum-of-squared-errors
    convention; if the data-fit term were mean squared error, the matching
    diagonal term would be ``n * lambda * I``. Use ``np.linalg.solve`` rather
    than explicitly inverting the matrix.

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        y (np.ndarray): Targets, shape (n,).
        alpha (float): Regularization strength, >= 0.

    Returns:
        np.ndarray: Parameters, shape (d,).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2 — Logistic regression from scratch
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Numerically stable logistic sigmoid, elementwise.

    sigmoid(z) = 1 / (1 + exp(-z))

    Must not overflow for large |z| (e.g. z = ±1000). Hint: treat the
    positive and negative branches separately (``np.where``) or use
    ``np.clip`` on z, so that ``exp`` is only ever called on non-positive
    (or bounded) arguments.

    Args:
        z (np.ndarray or float): Input value(s).

    Returns:
        np.ndarray or float: Sigmoid of ``z``, same shape, values in (0, 1)
            (0.0 / 1.0 at the saturated extremes is acceptable).
    """
    raise NotImplementedError


def logistic_gradient(X, y, w):
    """Gradient of the mean logistic (cross-entropy) loss w.r.t. ``w``.

    For labels y in {0, 1}:
        grad = (1/n) * X.T @ (sigmoid(X @ w) - y)

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        y (np.ndarray): Binary labels in {0, 1}, shape (n,).
        w (np.ndarray): Weights, shape (d,).

    Returns:
        np.ndarray: Gradient vector, shape (d,).
    """
    raise NotImplementedError


def predict_proba(X, w):
    """Predicted probability of class 1 for each row of ``X``.

    p = sigmoid(X @ w)

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        w (np.ndarray): Weights, shape (d,).

    Returns:
        np.ndarray: Probabilities in [0, 1], shape (n,).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3 — scikit-learn pipeline & cross-validation
# ---------------------------------------------------------------------------
def build_pipeline():
    """Build a leakage-proof classification pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A two-step pipeline:
            1. ``StandardScaler()``
            2. ``LogisticRegression(max_iter=1000)``
        Because scaling lives INSIDE the pipeline, cross-validation refits
        the scaler on each training fold only — no information from the
        validation fold leaks into preprocessing.
    """
    raise NotImplementedError


def evaluate_with_cv(pipeline, X, y):
    """Mean 5-fold cross-validated accuracy of a pipeline.

    Use ``sklearn.model_selection.cross_val_score`` with ``cv=5`` and
    ``scoring="accuracy"``, and return the mean of the fold scores.

    Args:
        pipeline (sklearn.pipeline.Pipeline): An (unfitted) estimator.
        X (np.ndarray): Features, shape (n, d).
        y (np.ndarray): Labels, shape (n,).

    Returns:
        float: Mean cross-validated accuracy, in [0, 1].
    """
    raise NotImplementedError
