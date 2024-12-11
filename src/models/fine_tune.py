import os

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import dataset
from src.models import trainer
from src.tools.startup import logger

_pipeline_name = 'fine_tune'


def run(_settings: dict) -> None:
    """
    Thins function runs a fine-tune process of pre-trained Transformer model.
    It extracts the train dataset from disk, splits it into train and
    validation sets. Also, it runs the fine-tune process. Finally, the
    tokenizer, the model and metrics are stored on disk.

    Args:
        _settings (dict): a dict with configuration settings.
    """
    fine_tune_settings = _settings[_pipeline_name]

    # Check if file path exists. Because it depends on the environment of the
    # execution. From Docker '/data/...'. Otherwise, Colab for example
    # './data'.
    train_filepath = fine_tune_settings['dataset']['filepath']
    if not os.path.exists(train_filepath):
        train_filepath = f'.{train_filepath}'

    # Extract train dataset.
    train_df = pd.read_csv(
        train_filepath, **fine_tune_settings['dataset']['params'])

    # Generate train and validation splits.
    train_df, val_df = train_test_split(
        train_df,
        stratify=train_df[fine_tune_settings['dataset']['target_column']],
        **fine_tune_settings['dataset']['train_test_split'])

    logger.info(f'Train size {train_df.shape}')
    logger.info(f'Validation size {val_df.shape}')

    # Datasets creation.
    train_dataset = dataset.FounderNestDataset(
        train_df, fine_tune_settings['dataset'])
    val_dataset = dataset.FounderNestDataset(
        val_df, fine_tune_settings['dataset'])

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(
        **fine_tune_settings['tokenizer']['from_pretrained'])

    train_dataset \
        .tokenize(tokenizer) \
        .convert_to_tensors()

    val_dataset \
        .tokenize(tokenizer) \
        .convert_to_tensors()

    # Create model instance.
    model = AutoModelForSequenceClassification.from_pretrained(
        **fine_tune_settings['transformer']['from_pretrained'],
        num_labels=train_dataset.num_categories)

    # Fine-tune model.
    trainer_obj = trainer.Trainer(_settings)

    trainer_obj.train(
        model,
        train_dataset,
        val_dataset,
        tokenizer)
