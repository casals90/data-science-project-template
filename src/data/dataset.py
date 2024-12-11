from typing import Dict

import pandas as pd
from torch.utils import data as torch_data

from src.models import utils as model_utils
from src.tools.startup import logger


class FounderNestDataset(torch_data.Dataset):
    def __init__(self, df: pd.DataFrame, _settings: Dict) -> None:
        super().__init__()

        self._df = df

        self._settings = _settings

        self._text_column = _settings['text_column']
        self._target_column = _settings['target_column']
        self._length = self._df.shape[0]

        self._encodings = None
        self._labels = None

    @property
    def num_categories(self) -> int:
        """
        Get the number of unique categories in the dataframe.

        Returns:
            (int): The number of dataset's categories.
        """
        return self._df[self._target_column].nunique()

    @property
    def class_weights(self) -> Dict[int, float]:
        """
        Compute the weight of each class in the dataframe.

        Returns:
            (Dict[int, ct.Number]): a dict with the weight for each class.
        """
        return model_utils.compute_class_weight(self._df[self._target_column])

    def tokenize(self, tokenizer):
        """
        This function runs a tokenizer in a text column and concat the result
        into the dataframe.

        Args:
            tokenizer: a tokenizer function to apply to text column.
        """
        logger.info(f"Running tokenize...")
        self._encodings = tokenizer(
            self._df[self._settings['text_column']].to_list(),
            **self._settings['tokenize'])
        logger.info('Done.')

        return self

    def convert_to_tensors(self):
        """
        This function converts to Tensor and the following data:
            - encodings: the results of tokenizer process.
            - labels: the labels of dataset.
            - extra features: the additional features of the dataset.
        """
        logger.info(f"Running convert_to_tensors...")
        self._labels = self._df[self._target_column].values
        logger.info('Done.')

        return self

    def __getitem__(self, idx: int) -> dict:
        """
        Given an index, this function prepares the 'i' sample as a Deep
        Learning model input generating a dict with input features.

        Args:
            idx (int): the index of the sample.

        Returns:
            (dict): a dict with the Deep Learning model's inputs.
        """
        item = {
            key: val[idx] for key, val in self._encodings.items()
        }
        item['labels'] = self._labels[idx]

        return item

    def __len__(self) -> int:
        """
        Get the length of dataframe.

        Returns:
            (int): the dataframe length.
        """
        return self._length
