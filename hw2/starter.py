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
    predictions = X @ theta  # multiply the matrix and parameters
    errors = predictions - y  # difference between predictions and actual target y
    loss = np.mean(errors ** 2)  # what is the mean of the errors, squared?  return that.

    return float(loss)  #cast the return as a float as requested. session 8 page 3

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
    #like above, but more.
    n = X.shape[0]    # n is the shape
    predictions = X @ theta      # predictions
    residual_vector = predictions - y     # residual vector, or "errors"
    sum_of_weighted_errors = X.T @ residual_vector    # sum of weighted errors
    grad = (2/n) * sum_of_weighted_errors   # gradient vector
    return grad    # session 8 pages 4 and 5   


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
    d = X.shape[1]    # d is the number of features
    theta = np.zeros(d)  # starting from theta = zeros.
    loss_history = []    # empty list of history
    for epoch in range(epochs):
        theta = theta - lr * mse_gradient(X, y, theta)    # move theta in opposite direction of gradient
        current_loss = mse_loss(X, y, theta)     # current loss with new theta
        loss_history.append(current_loss)     # keep history
    return theta, loss_history    # return the tuple of theta and the history, session 8 pages 6 and 7

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
   # return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
    d = X.shape[1]     # number of features
    I = np.identity(d)     # identity matrix of shape (d, d)
    A = X.T @ X + alpha * I     # matrix to be inverted
    b = X.T @ y       # right-hand side vector
    theta = np.linalg.solve(A, b)   # solve the linear system for theta using linalg.solve instead of inverting A (and introducing numeric instability)
    return theta    # session 9 pages 2 and 3

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
    z_clipped = np.clip(z, -500, 500)     # clip to avoid overflow
    sigmoid_of_z = 1 / (1 + np.exp(-z_clipped))    # numerically stable sigmoid of z
    return sigmoid_of_z              #session 10 pages 1 and 2

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
    # almost the same as above in the linear gradient
    n = X.shape[0]    # n is the shape
    predictions = sigmoid(X @ w)      # probabilities / predictions.  variable name is "w" for weights instead of "theta", but is still a 1D vector of shape (d,)
    residual_vector = predictions - y     # residual vector, or "errors"
    sum_of_weighted_errors = X.T @ residual_vector    # sum of weighted errors
    grad = (1/n) * sum_of_weighted_errors   # scale by 1/n to get mean gradient
    return grad   #  session 10 pages 4 and 5


def predict_proba(X, w):
    """Predicted probability of class 1 for each row of ``X``.

    p = sigmoid(X @ w)

    Args:
        X (np.ndarray): Design matrix, shape (n, d).
        w (np.ndarray): Weights, shape (d,).

    Returns:
        np.ndarray: Probabilities in [0, 1], shape (n,).
    """
    raw_model_outputs = X @ w    # linear combination of inputs and weights
    probabilities = sigmoid(raw_model_outputs)    # apply sigmoid to get probabilities
    return probabilities    # session 10 pages 5 and 6


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
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000))])  
    return pipeline   # session 11 page 10 and 11.

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
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    mean_accuracy = scores.mean()
    return float(mean_accuracy)   # session 11 page 10 and 11.
