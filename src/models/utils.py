from typing import Dict, Tuple

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


def model_step(model, batch, device) -> Tuple:
    """
    Given a model, batch and device, this function pass to the model the
    batch items in order to get the loss value and the 'logits' of the model.

    Args:
        model: the pretrained model to generate the outputs
        batch: a dict with batch items to use
        device: the device where the model is located

    Returns:
        (Tuple): the loss and the 'logits' of the model output.
    """
    # Send inputs to device
    for in_key in batch:
        batch[in_key] = batch[in_key].to(device)

    # Clear previously calculated gradients.
    model.zero_grad()
    # Run the model
    outputs = model(**batch)

    return outputs['loss'], outputs['logits']
