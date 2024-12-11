import math
from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as ss


# Correlation between categorical and numeric features

def compute_correlation_ratio(
        measurements: Union[pd.Series, np.ndarray], categories:
        Union[pd.Series, np.ndarray]) -> float:
    """
    Given a list of categorical dataframe columns and a list of numeric
    features, this function computes the ETA correlation.
    Args:
        measurements (Union[pd.Series, ct.Array]): a sequence of numeric 
            measurements.
        categories (Union[pd.Series, ct.Array]): a sequence of categorical 
            measurements.
    Notes:
        ETA correlation is useful for computing the correlation between numeric
        features and categorical.
        The source code function:
        https://stackoverflow.com/questions/52083501/how-to-compute-
        correlation-ratio-or-eta-in-python/52084418
    Returns:
        (float): the ETA correlation value
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(
        n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)

    return eta


# Correlation between categorical and categorical features

# Source: https://github.com/shakedzy/dython/tree/
# da87f99f9787ceac55e9b6281e9750d62a1ad0a1

def compute_cramers_v(
        x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) \
        -> float:
    """
    This function computes the Cramers V statistic for categorical to
    categorical association. In addition, the value is a range between [0, 1].

    Args:
        x (Union[pd.Series, ct.Array]): a sequence of categorical measurements.
        y (Union[pd.Series, ct.Array]): a sequence of categorical measurements.

    Returns:
        (float): Cramers V correlation value between x and y.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def conditional_entropy(
        x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray],
        log_base: float = math.e) -> float:
    """
    This function computes the conditional entropy of x given y: S(x|y)

    Args:
        x (Union[pd.Series, ct.Array]): a sequence of categorical measurements.
        y (Union[pd.Series, ct.Array]): a sequence of categorical measurements.
        log_base (optional, float): base for calculating entropy. The default
            value is e.

    Returns:
        (float): the conditional entropy value between x and y.
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def compute_theils_u(
        x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) \
        -> float:
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. In addition, the value is a range between [0, 1].

    Args:
        x (Union[pd.Series, ct.Array]): a sequence of categorical measurements.
        y (Union[pd.Series, ct.Array]): a sequence of categorical measurements.

    Returns:
        (float): the Theil V correlation between x and y.
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)

    if s_x == 0:
        result = 1
    else:
        result = (s_x - s_xy) / s_x

    return result
