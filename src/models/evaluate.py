import os
from typing import Dict
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

from src.models import extract, base
from src.tools.startup import logger

_pipeline_name = 'evaluate_from_pretrained'
_evaluate_arguments = 'evaluate_arguments'


def compute_classification_metrics(
        y_true: Union[pd.Series, np.ndarray], y_predicted:
        Union[pd.Series, np.ndarray], average: str = 'macro',
        flat: bool = False) -> Dict[str, np.ndarray]:
    """
    This function computes a classifier report dictionary containing most
    used classification metrics such as accuracy, precision, recall, etc.

    Args:
        y_true (Union[pd.Series, ct.Array]): series of ground truth classes.
        y_predicted (Union[pd.Series, ct.Array]): series of predicted classes.
        average (optional, str): the average parameter of sklearn metrics
            function. This is for the way how to compute multi-label
            classification. Default value is macro.
        flat (optional, bool): 

    Returns:
        (Dict[str, float]): dict containing computed metrics
    """
    if flat:
        y_predicted = np.argmax(y_predicted, axis=1).flatten()
        y_true = y_true.flatten()

    metrics = {
        'accuracy': sklearn_metrics.accuracy_score(y_true, y_predicted),
        'precision': sklearn_metrics.precision_score(
            y_true, y_predicted, average=average, zero_division=0),
        'sensitivity': sklearn_metrics.recall_score(
            y_true, y_predicted, average=average, zero_division=0),
        'f1': sklearn_metrics.f1_score(
            y_true, y_predicted, average=average, zero_division=0),
    }

    return metrics


def compute_detailed_metrics(
        y, predictions, labels: Union[List[str], List[int]] = None,
        output_dict: bool = False) -> Union[Dict, str]:
    """
   This function compute the performance of the model in the dataset.
   It computes the following classification metrics:
       - accuracy
       - precision
       - sensitivity
       - f1
   Finally, it computes the metrics for each dataset's category.

   Args:
       y (ct.Array): an array of targets.
       predictions (ct.Array): an array of predictions.
       labels (optional, ct.StrList): a list of target's labels to use
           in sklearn method 'classification_report'.
       output_dict (optional, bool): It is a parameter of sklearn method
           'classification_report'. It refers if the result of the
           function is a Dict or String. Default value is False.

   Returns:
       (Tuple[Dict[str, ct.Number], Union[Dict, str]]): a dict with
           average metrics and a Dict/String with metrics per class.
   """
    classification_report = sklearn_metrics.classification_report(
        y, predictions, labels=labels, zero_division=0,
        output_dict=output_dict)

    return classification_report


def get_train_test_dataset(
        dataset: pd.DataFrame, train_index: np.ndarray,
        test_index: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test based on index/position.

    Args:
        dataset (ct.Dataset): Input dataset.
        train_index (ct.Array): Train indexes.
        test_index (ct.Array): Dataset test indexes.

    Notes:
        Always select the train and test split based on position, for
        this reason when the dataset is a pd.DataFrame or pd.Series
        it uses the 'iloc' slicing (numeric position).
        On the other hand, with np.ndarray the normal slicing is fine.

    Returns:
        Train and test datasets.
    """
    if isinstance(dataset, (pd.DataFrame, pd.Series)):
        x_train, x_test = dataset.iloc[train_index], dataset.iloc[test_index]
    elif isinstance(dataset, np.ndarray):
        x_train, x_test = dataset[train_index], dataset[test_index]
    else:
        raise ValueError(
            f'Dataset type {type(dataset)} is not a valid')

    return x_train, x_test


class Evaluation(base.TransformerBase):
    def __init__(
            self, model_id: str, _settings: dict) -> None:
        super().__init__(_settings, _pipeline_name, _evaluate_arguments)

        self._model_id = model_id

        # Load settings, preprocessor, tokenizer and model.
        self._model_path = os.path.join(
            _settings['volumes']['models'], model_id)

        self._tokenizer, self._model = \
            extract.from_pretrained(self._model_path)

    @property
    def tokenizer(self) -> callable:
        return self._tokenizer

    def evaluate(self, dataset):
        loader = self._create_dataloader(
            dataset,
            self._args['sampler'],
            self._args['loader'])

        self._model.to(self._device)

        loss_list = []
        # Set model to evaluate mode.
        self._model.eval()
        predictions, labels = [], []
        for batch in tqdm(loader):
            with torch.set_grad_enabled(False):
                loss, logits = self._model_step(self._model, batch)
                loss_list.append(loss.item())

                batch_predictions = np.argmax(logits.cpu(), axis=1).flatten()
                predictions.extend(batch_predictions)
                labels.extend(batch['labels'].cpu())

        # Computes metrics.
        metrics = compute_classification_metrics(labels, predictions)
        classification_report = compute_detailed_metrics(
            labels, predictions, output_dict=False)
        logger.info(f'Metrics\n{metrics}')
        logger.info(f'classification_report\n{classification_report}')

        # Store all statistics of the epoch.
        stats = {
            'loss': np.mean(loss_list),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'sensitivity': metrics['sensitivity'],
            'f1': metrics['f1'],
        }
        self._report_metrics(stats)

        return metrics, classification_report

    def _report_metrics(
            self, metrics: Dict[str, float]) -> None:
        # Always 0, since it is evaluation.
        epoch = 0
        # Average validation loss
        logger.info(f'Loss {metrics["loss"]}')

        # Validation metrics
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        sensitivity = metrics['sensitivity']
        f1 = metrics['f1']

        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Sensitivity: {sensitivity}')
        logger.info(f'F1: {f1}')

        # Tensorboard logs
        self._writer.add_scalar(
            "Test/Metrics/Accuracy/", accuracy, epoch)
        self._writer.add_scalar(
            "Test/Metrics/Precision/", precision, epoch)
        self._writer.add_scalar(
            "Test/Metrics/Sensitivity/", sensitivity, epoch)
        self._writer.add_scalar("Test/Metrics/F1/", f1, epoch)
