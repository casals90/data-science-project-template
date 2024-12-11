from typing import Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.tools.startup import logger


def from_pretrained(model_path: str) \
        -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Given a local path, this function loads the pretrained tokenizer and
    pretrained model.

    Args:
        model_path (str): path to load tokenizer and model.

    Returns:
        (Tuple[AutoTokenizer, AutoModelForSequenceClassification]): the
        tokenizer and model.
    """
    logger.info(f'Extracting pretrained model from {model_path}')

    # Extract pretrained tokenizer and pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model
