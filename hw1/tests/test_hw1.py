"""HW1 autograder tests. Run from the hw1/ directory: pytest tests/ -v"""

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

HW_DIR = str(pathlib.Path(__file__).resolve().parents[1])

# Load starter.py under a unique module name so a repo-wide pytest run does
# not collide with other assignments' modules of the same file name.
_STARTER_PATH = pathlib.Path(HW_DIR) / "starter.py"
_spec = importlib.util.spec_from_file_location("hw1_starter", _STARTER_PATH)
starter = importlib.util.module_from_spec(_spec)
sys.modules["hw1_starter"] = starter
_spec.loader.exec_module(starter)

fill_missing_with_group_mean = starter.fill_missing_with_group_mean
merge_and_aggregate = starter.merge_and_aggregate
moving_average = starter.moving_average
pairwise_distances = starter.pairwise_distances
standardize = starter.standardize
tidy_wide_to_long = starter.tidy_wide_to_long
top_k_rows = starter.top_k_rows

DATA_DIR = os.path.join(HW_DIR, "data")


@pytest.fixture
def orders_df():
    return pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))


@pytest.fixture
def customers_df():
    return pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------
class TestStandardize:
    def test_columns_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
        Z = standardize(X)
        assert Z.shape == X.shape
        np.testing.assert_allclose(Z.mean(axis=0), np.zeros(4), atol=1e-10)
        np.testing.assert_allclose(Z.std(axis=0), np.ones(4), atol=1e-10)

    def test_exact_small_case(self):
        X = np.array([[1.0, 10.0], [3.0, 30.0]])
        Z = standardize(X)
        np.testing.assert_allclose(Z, np.array([[-1.0, -1.0], [1.0, 1.0]]))

    def test_input_not_mutated(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_copy = X.copy()
        standardize(X)
        np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------
class TestPairwiseDistances:
    def test_exact_small_case(self):
        A = np.array([[0.0, 0.0], [3.0, 4.0]])
        B = np.array([[0.0, 0.0], [6.0, 8.0], [3.0, 0.0]])
        D = pairwise_distances(A, B)
        expected = np.array([[0.0, 10.0, 3.0], [5.0, 5.0, 4.0]])
        assert D.shape == (2, 3)
        np.testing.assert_allclose(D, expected)

    def test_against_reference_loop(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(7, 5))
        B = rng.normal(size=(9, 5))
        D = pairwise_distances(A, B)
        expected = np.array(
            [[np.linalg.norm(a - b) for b in B] for a in A]
        )
        assert D.shape == (7, 9)
        np.testing.assert_allclose(D, expected, atol=1e-10)

    def test_self_distance_zero_diagonal(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 3))
        D = pairwise_distances(A, A)
        np.testing.assert_allclose(np.diag(D), np.zeros(6), atol=1e-10)


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------
class TestMovingAverage:
    def test_window_two(self):
        out = moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        np.testing.assert_allclose(out, [1.5, 2.5, 3.5])

    def test_window_three(self):
        out = moving_average(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), 3)
        np.testing.assert_allclose(out, [4.0, 6.0, 8.0])

    def test_window_one_identity(self):
        x = np.array([5.0, -1.0, 2.0])
        np.testing.assert_allclose(moving_average(x, 1), x)

    def test_window_full_length(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = moving_average(x, 4)
        assert len(out) == 1
        np.testing.assert_allclose(out, [2.5])

    def test_output_length(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=50)
        assert len(moving_average(x, 7)) == 44


# ---------------------------------------------------------------------------
# top_k_rows
# ---------------------------------------------------------------------------
class TestTopKRows:
    def test_basic(self):
        df = pd.DataFrame({"a": [10, 40, 20, 30], "b": ["w", "x", "y", "z"]})
        out = top_k_rows(df, "a", 2)
        assert list(out["a"]) == [40, 30]
        assert list(out["b"]) == ["x", "z"]

    def test_index_preserved(self):
        df = pd.DataFrame({"a": [1, 5, 3]}, index=[10, 11, 12])
        out = top_k_rows(df, "a", 2)
        assert list(out.index) == [11, 12]

    def test_k_equals_len(self):
        df = pd.DataFrame({"a": [2, 1, 3]})
        out = top_k_rows(df, "a", 3)
        assert list(out["a"]) == [3, 2, 1]

    def test_input_not_mutated(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        before = df.copy()
        top_k_rows(df, "a", 1)
        pd.testing.assert_frame_equal(df, before)

    def test_on_orders_csv(self, orders_df):
        out = top_k_rows(orders_df, "quantity", 2)
        assert list(out["order_id"]) == [102, 113]


# ---------------------------------------------------------------------------
# fill_missing_with_group_mean
# ---------------------------------------------------------------------------
class TestFillMissingWithGroupMean:
    def test_basic_fill(self):
        df = pd.DataFrame(
            {
                "g": ["a", "a", "a", "b", "b"],
                "v": [1.0, np.nan, 3.0, 10.0, np.nan],
            }
        )
        out = fill_missing_with_group_mean(df, "g", "v")
        np.testing.assert_allclose(out["v"].to_numpy(), [1.0, 2.0, 3.0, 10.0, 10.0])

    def test_non_missing_untouched(self):
        df = pd.DataFrame({"g": ["a", "b"], "v": [1.5, 2.5]})
        out = fill_missing_with_group_mean(df, "g", "v")
        np.testing.assert_allclose(out["v"].to_numpy(), [1.5, 2.5])

    def test_all_nan_group_stays_nan(self):
        df = pd.DataFrame(
            {"g": ["a", "a", "b"], "v": [np.nan, np.nan, 7.0]}
        )
        out = fill_missing_with_group_mean(df, "g", "v")
        assert out["v"].isna().tolist() == [True, True, False]

    def test_input_not_mutated_and_new_frame(self):
        df = pd.DataFrame({"g": ["a", "a"], "v": [1.0, np.nan]})
        before = df.copy()
        out = fill_missing_with_group_mean(df, "g", "v")
        pd.testing.assert_frame_equal(df, before)
        assert out is not df

    def test_shape_and_columns_preserved(self):
        df = pd.DataFrame(
            {"g": ["a", "b", "a"], "v": [np.nan, 2.0, 4.0], "other": [7, 8, 9]}
        )
        out = fill_missing_with_group_mean(df, "g", "v")
        assert list(out.columns) == ["g", "v", "other"]
        assert out.shape == df.shape


# ---------------------------------------------------------------------------
# merge_and_aggregate
# ---------------------------------------------------------------------------
class TestMergeAndAggregate:
    def test_columns_and_order(self, orders_df, customers_df):
        out = merge_and_aggregate(orders_df, customers_df)
        assert list(out.columns) == [
            "customer_id",
            "name",
            "n_orders",
            "total_spent",
        ]
        assert list(out.index) == list(range(len(out)))

    def test_values_on_csv_data(self, orders_df, customers_df):
        out = merge_and_aggregate(orders_df, customers_df)
        assert len(out) == 6
        assert list(out["customer_id"]) == [1, 2, 4, 5, 3, 6]
        assert list(out["name"]) == [
            "Ava Nguyen",
            "Ben Ortiz",
            "Dev Patel",
            "Elle Johnson",
            "Cara Smith",
            "Finn Murphy",
        ]
        assert list(out["n_orders"]) == [4, 3, 3, 3, 3, 2]
        np.testing.assert_allclose(
            out["total_spent"].to_numpy(),
            [79.99, 74.99, 68.49, 62.00, 52.74, 25.50],
        )

    def test_inner_merge_drops_orderless_customers(self):
        orders = pd.DataFrame(
            {
                "order_id": [1, 2],
                "customer_id": [10, 10],
                "product": ["x", "y"],
                "quantity": [1, 2],
                "unit_price": [3.0, 4.0],
            }
        )
        customers = pd.DataFrame(
            {
                "customer_id": [10, 99],
                "name": ["Has Orders", "No Orders"],
                "state": ["ID", "WA"],
            }
        )
        out = merge_and_aggregate(orders, customers)
        assert len(out) == 1
        assert out.loc[0, "name"] == "Has Orders"
        assert out.loc[0, "n_orders"] == 2
        assert out.loc[0, "total_spent"] == pytest.approx(11.0)

    def test_inputs_not_mutated(self, orders_df, customers_df):
        o_before = orders_df.copy()
        c_before = customers_df.copy()
        merge_and_aggregate(orders_df, customers_df)
        pd.testing.assert_frame_equal(orders_df, o_before)
        pd.testing.assert_frame_equal(customers_df, c_before)


# ---------------------------------------------------------------------------
# tidy_wide_to_long
# ---------------------------------------------------------------------------
class TestTidyWideToLong:
    def test_shape_and_columns(self):
        wide = pd.DataFrame(
            {
                "city": ["Moscow", "Boise"],
                "jan": [-2.0, 0.5],
                "feb": [1.0, 3.5],
                "mar": [6.0, 8.0],
            }
        )
        long = tidy_wide_to_long(wide)
        assert list(long.columns) == ["city", "month", "value"]
        assert len(long) == 6

    def test_values(self):
        wide = pd.DataFrame(
            {"city": ["A", "B"], "jan": [1.0, 2.0], "feb": [3.0, 4.0]}
        )
        long = tidy_wide_to_long(wide)
        lookup = {
            (row.city, row.month): row.value for row in long.itertuples()
        }
        assert lookup == {
            ("A", "jan"): 1.0,
            ("B", "jan"): 2.0,
            ("A", "feb"): 3.0,
            ("B", "feb"): 4.0,
        }

    def test_input_not_mutated(self):
        wide = pd.DataFrame({"city": ["A"], "jan": [1.0]})
        before = wide.copy()
        tidy_wide_to_long(wide)
        pd.testing.assert_frame_equal(wide, before)
