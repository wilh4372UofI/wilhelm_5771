"""HW2 autograder tests. Run from the hw2/ directory: pytest tests/ -v"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load starter.py under a unique module name so a repo-wide pytest run does
# not collide with other assignments' modules of the same file name.
_STARTER_PATH = pathlib.Path(__file__).resolve().parents[1] / "starter.py"
_spec = importlib.util.spec_from_file_location("hw2_starter", _STARTER_PATH)
starter = importlib.util.module_from_spec(_spec)
sys.modules["hw2_starter"] = starter
_spec.loader.exec_module(starter)

build_pipeline = starter.build_pipeline
evaluate_with_cv = starter.evaluate_with_cv
gradient_descent = starter.gradient_descent
logistic_gradient = starter.logistic_gradient
mse_gradient = starter.mse_gradient
mse_loss = starter.mse_loss
predict_proba = starter.predict_proba
ridge_closed_form = starter.ridge_closed_form
sigmoid = starter.sigmoid


def make_regression_data(seed=0, n=100, d=3, noise=0.1):
    """Well-conditioned synthetic linear-regression data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    theta_true = np.array([2.0, -1.0, 0.5])[:d]
    y = X @ theta_true + noise * rng.normal(size=n)
    return X, y, theta_true


def make_classification_data(seed=1, n=200, d=2):
    """Linearly separable-ish binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([3.0, -2.0])[:d]
    p = 1.0 / (1.0 + np.exp(-(X @ w_true)))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def numerical_gradient(f, w, eps=1e-6):
    """Central-difference numerical gradient of scalar function f at w."""
    grad = np.zeros_like(w, dtype=float)
    for i in range(w.size):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        grad[i] = (f(w_plus) - f(w_minus)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# mse_loss
# ---------------------------------------------------------------------------
class TestMseLoss:
    def test_zero_loss_at_true_theta_no_noise(self):
        X, y, theta_true = make_regression_data(noise=0.0)
        assert float(mse_loss(X, y, theta_true)) == pytest.approx(0.0, abs=1e-12)

    def test_exact_small_case(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([1.0, 2.0])
        theta = np.array([0.0, 0.0])
        # residuals (-1, -2) -> mean of (1, 4) = 2.5
        assert float(mse_loss(X, y, theta)) == pytest.approx(2.5)

    def test_matches_reference(self):
        X, y, _ = make_regression_data(seed=4)
        theta = np.array([1.0, 1.0, 1.0])
        expected = np.mean((X @ theta - y) ** 2)
        assert float(mse_loss(X, y, theta)) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# mse_gradient (incl. gradient check)
# ---------------------------------------------------------------------------
class TestMseGradient:
    def test_shape(self):
        X, y, _ = make_regression_data()
        g = mse_gradient(X, y, np.zeros(3))
        assert g.shape == (3,)

    def test_zero_at_optimum_no_noise(self):
        X, y, theta_true = make_regression_data(noise=0.0)
        g = mse_gradient(X, y, theta_true)
        np.testing.assert_allclose(g, np.zeros(3), atol=1e-10)

    def test_numerical_gradient_check(self):
        X, y, _ = make_regression_data(seed=5)
        theta = np.array([0.3, -0.7, 1.2])
        analytical = mse_gradient(X, y, theta)
        numerical = numerical_gradient(lambda t: float(mse_loss(X, y, t)), theta)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# gradient_descent
# ---------------------------------------------------------------------------
class TestGradientDescent:
    def test_returns_theta_and_history(self):
        X, y, _ = make_regression_data()
        theta, history = gradient_descent(X, y, lr=0.1, epochs=50)
        assert theta.shape == (3,)
        assert len(history) == 50

    def test_loss_history_non_increasing(self):
        X, y, _ = make_regression_data(seed=6)
        _, history = gradient_descent(X, y, lr=0.05, epochs=100)
        history = np.asarray(history, dtype=float)
        assert np.all(np.diff(history) <= 1e-12)

    def test_converges_to_true_theta(self):
        X, y, theta_true = make_regression_data(seed=7, noise=0.0)
        theta, history = gradient_descent(X, y, lr=0.1, epochs=500)
        np.testing.assert_allclose(theta, theta_true, atol=1e-3)
        assert history[-1] == pytest.approx(0.0, abs=1e-6)

    def test_final_loss_below_initial(self):
        X, y, _ = make_regression_data(seed=8)
        _, history = gradient_descent(X, y, lr=0.05, epochs=100)
        initial_loss = float(mse_loss(X, y, np.zeros(3)))
        assert history[-1] < initial_loss


# ---------------------------------------------------------------------------
# ridge_closed_form
# ---------------------------------------------------------------------------
class TestRidgeClosedForm:
    def test_alpha_zero_recovers_ols(self):
        X, y, theta_true = make_regression_data(seed=9, noise=0.0)
        theta = ridge_closed_form(X, y, alpha=0.0)
        np.testing.assert_allclose(theta, theta_true, atol=1e-8)

    def test_matches_normal_equation(self):
        X, y, _ = make_regression_data(seed=10)
        alpha = 2.5
        expected = np.linalg.solve(
            X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y
        )
        np.testing.assert_allclose(
            ridge_closed_form(X, y, alpha), expected, atol=1e-10
        )

    def test_shrinkage(self):
        X, y, _ = make_regression_data(seed=11)
        small = ridge_closed_form(X, y, alpha=0.0)
        large = ridge_closed_form(X, y, alpha=1000.0)
        assert np.linalg.norm(large) < np.linalg.norm(small)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------
class TestSigmoid:
    def test_midpoint(self):
        assert float(sigmoid(0.0)) == pytest.approx(0.5)

    def test_known_values(self):
        assert float(sigmoid(2.0)) == pytest.approx(1 / (1 + np.exp(-2.0)))
        assert float(sigmoid(-2.0)) == pytest.approx(1 / (1 + np.exp(2.0)))

    def test_symmetry(self):
        z = np.array([-3.0, -1.0, 0.5, 4.0])
        np.testing.assert_allclose(sigmoid(z) + sigmoid(-z), np.ones(4), atol=1e-12)

    def test_numerically_stable_no_overflow(self):
        z = np.array([-1000.0, -100.0, 100.0, 1000.0])
        with np.errstate(over="raise"):
            out = sigmoid(z)
        out = np.asarray(out, dtype=float)
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(0.0, abs=1e-12)
        assert out[-1] == pytest.approx(1.0, abs=1e-12)

    def test_elementwise_shape(self):
        z = np.linspace(-5, 5, 11)
        assert np.asarray(sigmoid(z)).shape == (11,)


# ---------------------------------------------------------------------------
# logistic_gradient (incl. gradient check)
# ---------------------------------------------------------------------------
class TestLogisticGradient:
    def test_shape(self):
        X, y = make_classification_data()
        g = logistic_gradient(X, y, np.zeros(2))
        assert g.shape == (2,)

    def test_matches_reference_formula(self):
        X, y = make_classification_data(seed=12)
        w = np.array([0.5, -0.25])
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        expected = X.T @ (p - y) / len(y)
        np.testing.assert_allclose(logistic_gradient(X, y, w), expected, atol=1e-10)

    def test_numerical_gradient_check(self):
        X, y = make_classification_data(seed=13, n=60)
        w = np.array([0.4, -0.9])

        def nll(w_):
            p = 1.0 / (1.0 + np.exp(-(X @ w_)))
            eps = 1e-12
            return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        analytical = logistic_gradient(X, y, w)
        numerical = numerical_gradient(nll, w)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------
class TestPredictProba:
    def test_range_and_shape(self):
        X, _ = make_classification_data(seed=14)
        w = np.array([1.0, -1.0])
        p = predict_proba(X, w)
        assert p.shape == (len(X),)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)

    def test_matches_sigmoid_of_scores(self):
        X, _ = make_classification_data(seed=15, n=20)
        w = np.array([0.7, 0.2])
        expected = 1.0 / (1.0 + np.exp(-(X @ w)))
        np.testing.assert_allclose(predict_proba(X, w), expected, atol=1e-10)

    def test_zero_weights_give_half(self):
        X, _ = make_classification_data(seed=16, n=10)
        np.testing.assert_allclose(predict_proba(X, np.zeros(2)), 0.5 * np.ones(10))


# ---------------------------------------------------------------------------
# build_pipeline / evaluate_with_cv
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_type_composition(self):
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)
        steps = [step for _, step in pipe.steps]
        assert len(steps) == 2
        assert isinstance(steps[0], StandardScaler)
        assert isinstance(steps[1], LogisticRegression)

    def test_fresh_unfitted_pipeline(self):
        pipe = build_pipeline()
        # An unfitted scaler has no learned statistics yet.
        assert not hasattr(pipe.steps[0][1], "mean_")

    def test_cv_output_range(self):
        X, y = make_classification_data(seed=17, n=150)
        acc = evaluate_with_cv(build_pipeline(), X, y)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_cv_beats_chance_on_separable_data(self):
        X, y = make_classification_data(seed=18, n=300)
        acc = evaluate_with_cv(build_pipeline(), X, y)
        assert acc > 0.8

    def test_cv_is_mean_of_five_folds(self):
        from sklearn.model_selection import cross_val_score

        X, y = make_classification_data(seed=19, n=200)
        expected = cross_val_score(
            build_pipeline(), X, y, cv=5, scoring="accuracy"
        ).mean()
        acc = evaluate_with_cv(build_pipeline(), X, y)
        assert acc == pytest.approx(expected, abs=1e-12)
