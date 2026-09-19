"""HW1 — NumPy & Pandas: starter code.

Implement every function below by replacing ``raise NotImplementedError``.
Do not change any function signatures. Where a docstring says "no loops",
your solution must not use ``for``/``while`` loops or comprehensions over
array elements — use NumPy vectorization/broadcasting instead.

Run the tests from the hw1/ directory:
    pytest tests/ -v
"""

import numpy as np  # noqa: F401  (provided for student solutions)
import pandas as pd  # noqa: F401  (provided for student solutions)
from scipy import stats

# Chris Wilhelm - CS5722 - Homework 1 - 2026-09-19

# ---------------------------------------------------------------------------
# Part 1 — NumPy
# ---------------------------------------------------------------------------
def standardize(X):
    """Z-score each COLUMN of a 2-D array. No loops.

    Each column is transformed to (column - column_mean) / column_std,
    using the population standard deviation (``ddof=0``, NumPy's default).

    Args:
        X (np.ndarray): Array of shape (n_samples, n_features). You may
            assume every column has nonzero standard deviation.

    Returns:
        np.ndarray: Array of the same shape where every column has mean 0
            and standard deviation 1. Must not modify ``X`` in place.
    """
    answermatrix = stats.zscore(X, axis=0, nan_policy='omit')  # zscore an ndarray "X" on axis=0 (columns) and omit NaN, store in "answermatrix"
    return answermatrix  # return the answermatrix ndarray


def pairwise_distances(A, B):
    """Euclidean distance between every row of A and every row of B.

    Broadcasting only: no Python loops, no scipy. Hint: insert a new axis
    so that ``A[:, None, :] - B[None, :, :]`` has shape (n, m, d), then
    reduce over the last axis.

    Args:
        A (np.ndarray): Array of shape (n, d).
        B (np.ndarray): Array of shape (m, d).

    Returns:
        np.ndarray: Array ``D`` of shape (n, m) where
            ``D[i, j] = ||A[i] - B[j]||_2``.
    """
    # Euclidiean Distance is the sqrt of the sum of the elements subtracted, squared
    # using the hint given above

    differenceofelements = A[:, None, :] - B[None, :, :]  #difference
    squareddiffs = differenceofelements ** 2  #squared
    sumofsquarediffs = np.sum(squareddiffs, axis=-1)  #sum  -  last axis is -1 axis
    return np.sqrt(sumofsquarediffs) #return sqrt -- Euclidean Distance



def moving_average(x, k):
    """Moving average of a 1-D array with window size ``k``.

    Vectorized — no Python loop over the elements of ``x``. Hints:
    ``np.convolve(x, np.ones(k)/k, mode="valid")`` or a cumulative-sum
    trick both work.

    Args:
        x (np.ndarray): 1-D array of length n, with n >= k >= 1.
        k (int): Window size.

    Returns:
        np.ndarray: 1-D array of length ``n - k + 1`` where element ``i``
            is the mean of ``x[i : i + k]``.

    Example:
        >>> moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        array([1.5, 2.5, 3.5])
    """
    cumulative_sum = np.cumsum(x)  #cumulative sum of the array
    window_sum = cumulative_sum[k-1:].copy()  #copy the array to the point the window ends to a new array.
    window_sum[1:] -= cumulative_sum[:-k]  #subtract the cumulative sum of the original array up to the point the window begins from the new array.  You now have just the window.
    return (window_sum / k) #divide the window sum by the window size to get the average in the window


# ---------------------------------------------------------------------------
# Part 2 — Pandas
# ---------------------------------------------------------------------------
def top_k_rows(df, col, k):
    """Return the ``k`` rows with the largest values in column ``col``.

    Rows are ordered by ``col`` descending. Preserve the original index
    (do not reset it). ``df.nlargest`` is a good fit.

    Args:
        df (pd.DataFrame): Input frame.
        col (str): Name of a numeric column.
        k (int): Number of rows to return (assume 1 <= k <= len(df)).

    Returns:
        pd.DataFrame: ``k`` rows sorted by ``col`` descending. The input
            ``df`` must not be modified.
    """
    if (1 <= k) and (k <= len(df)):  #make sure k is what is assumed
        return df.nlargest(k, columns=col)   #per the hint, df.nlargest sorts descending and preserves original index by default
    


def fill_missing_with_group_mean(df, group_col, value_col):
    """Fill NaNs in ``value_col`` with the mean of the row's group.

    For each row where ``value_col`` is NaN, replace it with the mean of
    ``value_col`` over rows sharing the same ``group_col`` value (mean
    computed ignoring NaNs). If a group is entirely NaN, its values stay
    NaN. Hint: ``groupby(...)[value_col].transform("mean")``.

    Args:
        df (pd.DataFrame): Input frame.
        group_col (str): Column to group by.
        value_col (str): Numeric column that may contain NaNs.

    Returns:
        pd.DataFrame: A NEW frame (same shape, same index, same columns)
            with ``value_col`` filled. The input ``df`` must not be
            modified.
    """

    df_mycopy = df.copy()   # make a copy of the original df input
    group_averages = df_mycopy.groupby(group_col)[value_col].transform("mean")   #average value_col for each group.
    df_mycopy[value_col] = df_mycopy[value_col].fillna(group_averages)  #fill in the missing values with the average
    return df_mycopy


def merge_and_aggregate(orders_df, customers_df):
    """Merge orders with customers, then aggregate per customer.

    Steps:
      1. Compute each order's revenue: ``quantity * unit_price``.
      2. Inner-merge with ``customers_df`` on ``customer_id``.
      3. Group by ``customer_id`` and ``name``; aggregate:
         - ``n_orders``: number of orders (count of ``order_id``)
         - ``total_spent``: sum of revenue
      4. Return a frame with exactly the columns
         ``["customer_id", "name", "n_orders", "total_spent"]``,
         sorted by ``total_spent`` descending, index reset to 0..n-1.

    Args:
        orders_df (pd.DataFrame): Columns ``order_id, customer_id,
            product, quantity, unit_price``.
        customers_df (pd.DataFrame): Columns ``customer_id, name, state``.

    Returns:
        pd.DataFrame: Aggregated per-customer frame as specified above.
            Customers with no orders are excluded (inner merge). The
            inputs must not be modified.
    """
    df_myorders = orders_df.copy() # make a copy to not modify orders_df
    df_myorders["revenue"] = df_myorders["quantity"] * df_myorders["unit_price"]  #  1. calculate each order's revenue
    df_mymerge = pd.merge(df_myorders, customers_df, on="customer_id", how="inner") # 2. inner merge customers_df on customer_id
    df_agg = df_mymerge.groupby(["customer_id", "name"]).agg(n_orders=("order_id", "count"), total_spent=("revenue", "sum"))  # 3. group by customer_id and name, then aggregate
    df_answerframe = (df_agg.reset_index().sort_values(by="total_spent", ascending=False).reset_index(drop=True)) # 4. reset index gets rid of customer_id and name as the index and moves them back to just columns, sort by total_spent, descending.  Reset the index again to drop the index and recreate because we just sorted.
    return df_answerframe[["customer_id", "name", "n_orders", "total_spent"]]  # 4. return the frame with the specific columns requested

def tidy_wide_to_long(df):
    """Reshape a wide measurement table into tidy long format with melt.

    The input has one identifier column, ``city``, and any number of
    additional columns, one per month (e.g. ``jan``, ``feb``, ...). Use
    ``pd.melt`` with:
      - ``id_vars="city"``
      - ``var_name="month"``
      - ``value_name="value"``

    Args:
        df (pd.DataFrame): Wide frame with a ``city`` column plus one
            column per month.

    Returns:
        pd.DataFrame: Long frame with exactly the columns
            ``["city", "month", "value"]`` and default RangeIndex, as
            produced by ``pd.melt``. The input must not be modified.
    """
    return pd.melt(df, id_vars="city", var_name="month", value_name="value" )  #melting makes a copy, so df isn't modified.  return the "melted" table as requested