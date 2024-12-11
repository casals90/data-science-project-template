import os

import pandas as pd

from src.data import dataset
from src.models import evaluate

_pipeline_name = 'evaluate_from_pretrained'


def run(_settings) -> None:
    """
    This function computes the performance of the fine-tuned model. It extracts
    the test set and runs the evaluation process. The computed evaluation
    metrics are the following:
    - accuracy
    - precision
    - recall
    - f1 score

    Finally, a classification report for both classes are computed.

    Args:
        _settings (dict): a dict with configuration settings.
    """
    evaluate_settings = _settings[_pipeline_name]

    # Extract model id.
    model_id = evaluate_settings['extract']['model']

    # Check if file path exists. Because it depends on the environment of the
    # execution. From Docker '/data/...'. Otherwise, Colab for example
    # './data'.
    test_filepath = evaluate_settings['extract']['dataset']['filepath']
    if not os.path.exists(test_filepath):
        test_filepath = f'.{test_filepath}'
    # Extract test dataset.
    df = pd.read_csv(
        test_filepath, **evaluate_settings['extract']['dataset']['params'])

    validation_obj = evaluate.Evaluation(model_id, _settings)

    test_dataset = dataset.FounderNestDataset(
        df, evaluate_settings['extract']['dataset'])
    test_dataset \
        .tokenize(validation_obj.tokenizer) \
        .convert_to_tensors()

    return validation_obj.evaluate(test_dataset)
