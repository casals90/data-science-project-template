from typing import Dict

import numpy as np
from sklearn.utils import class_weight as sk_learn_class_weight


def compute_class_weight(
        y_train: np.ndarray, class_weight: str = 'balanced') \
        -> Dict[int, float]:
    """
    Given an array with target values, this function computes the weight of
    each class.

    Args:
        y_train (ct.Array): an array with target values.
        class_weight (optional, str): the strategy to compute class weight.

    Returns:
        (Dict[int, ct.Number]): a dict with the weight for each class.
    """
    classes = np.unique(y_train)
    class_weight = sk_learn_class_weight.compute_class_weight(
        class_weight=class_weight, classes=classes, y=y_train)

    return {c: w for c, w in zip(classes, class_weight)}
