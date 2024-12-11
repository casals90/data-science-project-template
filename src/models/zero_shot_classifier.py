from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import transformers
from tqdm import tqdm

from src.models import evaluate
from src.tools.startup import logger


class ZeroShotClassifier:
    def __init__(self, _settings: dict):
        self._settings = _settings

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        logger.info(f'Device: {self._device}')

    def _set_up(self):
        return transformers.pipeline(**self._settings)

    def predict(self, input_texts: List[str], class_descriptions: List[str]):
        zero_shot_classifier = self._set_up()
        predictions = []
        with tqdm(input_texts, unit="iter", desc=f'Predicting') as pbar:
            for input_text in pbar:
                iter_prediction = zero_shot_classifier(
                    input_text, class_descriptions)
                predictions.append(iter_prediction)

        return predictions

    def evaluate(
            self, x, y, class_descriptions: List[str],
            average: Optional[str] = 'macro') \
            -> Tuple[Dict[str, np.ndarray], Union[Dict, str], List[dict]]:
        predictions = self.predict(x, class_descriptions)
        scores = np.array([pr['scores'] for pr in predictions])

        scores = np.argmax(scores, axis=1)
        metrics = evaluate.compute_classification_metrics(y, scores, average)
        classification_report = evaluate.compute_detailed_metrics(y, scores)

        return metrics, classification_report, predictions
