import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models import evaluate, base
from src.tools import utils as tools_utils
from src.tools.startup import logger

_pipeline_name = 'fine_tune'
_trainer_arguments = 'trainer_arguments'


class Trainer(base.TransformerBase):
    def __init__(self, _settings) -> None:
        super().__init__(_settings, _pipeline_name, _trainer_arguments)

        self._set_seeds(_settings[_pipeline_name]['random_seed'])
        self._training_stats = []

    @staticmethod
    def _set_seeds(seed_value: int) -> None:
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def train(
            self,
            model,
            train_dataset,
            val_dataset,
            tokenizer):

        # Prepare train and validation data loaders.
        train_loader = self._create_dataloader(
            train_dataset,
            self._args['train_sampler'],
            self._args['train_loader'])

        val_loader = self._create_dataloader(
            val_dataset,
            self._args['eval_sampler'],
            self._args['eval_loader'])

        # Initialize optimizer.
        optimizer = AdamW(
            model.parameters(),
            **self._args['optim'])

        # Initialize learning rate scheduler.
        num_training_steps = len(train_loader) * self._args['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=num_training_steps,
            **self._args['scheduler'])

        # Send model to available device.
        model.to(self._device)

        # Training loop
        for epoch in range(self._args['epochs']):
            logger.info(f'Epoch {epoch}')
            epoch_stats = self._run_epoch(
                epoch, train_loader, val_loader, model, optimizer, scheduler)

            self._training_stats.append(epoch_stats)
            self._report_metrics(epoch_stats)

        self._writer.close()

        # Save the model.
        self._save(model, tokenizer)

        return self

    def _train_one_epoch(
            self, train_loader, model, optimizer, scheduler) \
            -> Tuple[float, str]:
        # Set model to train mode.
        model.train()

        train_loss = []
        with tqdm(train_loader, unit="batch", desc=f'Running train phase') \
                as train_pbar:
            t0 = time.time()
            for batch in train_pbar:
                with torch.set_grad_enabled(True):
                    loss, logits = self._model_step(model, batch)
                    train_loss.append(loss.item())

                    loss.backward()
                    # Clip the norm of the gradients to 1.0. This
                    # is to help prevent the "exploding gradients"
                    # problem.
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)

                    # Update parameters and take a step using the
                    # computed gradient. The optimizer dictates the
                    # "update rule"--how the parameters are
                    # modified based on their gradients, the
                    # learning rate, etc.
                    optimizer.step()

                    # Update the learning rate.
                    scheduler.step()

                # Adding loss to progress bar.
                progress_bar_metrics = {
                    'avg_loss': np.mean(train_loss)
                }
                train_pbar.set_postfix(progress_bar_metrics)

        # Compute the train epoch time.
        epoch_time = tools_utils.format_time(time.time() - t0)
        logger.info("Total training took {:} (h:mm:ss)".format(epoch_time))

        return np.mean(train_loss), epoch_time

    def _run_epoch(
            self, epoch, train_loader, val_loader, model, optimizer,
            scheduler) -> Dict[str, float]:
        # Train phase.
        avg_train_loss, train_time = self._train_one_epoch(
            train_loader, model, optimizer, scheduler)

        # Validation phase.
        val_loss, val_metrics = [], []
        # Set model to evaluate mode.
        model.eval()
        with tqdm(val_loader, unit="batch", desc=f'Running validation phase') \
                as val_pbar:
            t0 = time.time()
            for batch in val_pbar:
                with torch.set_grad_enabled(False):
                    loss, logits = self._model_step(model, batch)
                val_loss.append(loss.item())

                # Calculate the metrics of current validation batch and
                # accumulate it over all batches.
                batch_metrics = \
                    evaluate.compute_classification_metrics(
                        batch['labels'].cpu(), logits.cpu(), flat=True)
                val_metrics.append(batch_metrics)

                # Adding F1 to progress bar during val phase.
                progress_bar_metrics = {
                    'avg_loss': np.mean(val_loss),
                    'f1': batch_metrics['f1']
                }
                val_pbar.set_postfix(progress_bar_metrics)

        # Compute the validation epoch time.
        val_time = tools_utils.format_time(time.time() - t0)
        logger.info("Total validation took {:} (h:mm:ss)".format(val_time))

        # Compute the mean metrics of the validation pase.
        val_metrics_df = pd.DataFrame(val_metrics).mean()

        # Store all train statistics of the epoch.
        epoch_stats = {
            'epoch': epoch,
            'epoch_time': train_time,
            'training_loss': avg_train_loss,
            'validation_loss': np.mean(val_loss),
            'validation_accuracy': val_metrics_df['accuracy'],
            'validation_precision': val_metrics_df['precision'],
            'validation_sensitivity': val_metrics_df['sensitivity'],
            'validation_f1': val_metrics_df['f1'],
        }

        return epoch_stats

    def _report_metrics(self, epoch_stats: Dict[str, float]) -> None:
        epoch = int(epoch_stats['epoch'])
        logger.info("Training epoch took {:} (h:mm:ss)".format(
            epoch_stats['epoch_time']))

        # Average training loss
        logger.info(f'Epoch {epoch} average train loss '
                    f'{epoch_stats["training_loss"]}')
        self._writer.add_scalar(
            "Loss/train", epoch_stats['training_loss'], epoch)

        # Average validation loss
        logger.info(f'Epoch {epoch} average validation '
                    f'loss {epoch_stats["validation_loss"]}')
        self._writer.add_scalar(
            "Loss/validation", epoch_stats['validation_loss'], epoch)

        # Validation metrics
        accuracy = epoch_stats['validation_accuracy']
        precision = epoch_stats['validation_precision']
        sensitivity = epoch_stats['validation_sensitivity']
        f1 = epoch_stats['validation_f1']

        logger.info(f'Validation accuracy: {accuracy}')
        logger.info(f'Validation precision: {precision}')
        logger.info(f'Validation sensitivity: {sensitivity}')
        logger.info(f'Validation f1: {f1}')

        # Tensorboard logs
        self._writer.add_scalar(
            "Validation/Metrics/Accuracy/", accuracy, epoch)
        self._writer.add_scalar(
            "Validation/Metrics/Precision/", precision, epoch)
        self._writer.add_scalar(
            "Validation/Metrics/Sensitivity", sensitivity, epoch)
        self._writer.add_scalar("Validation/Metrics/F1/", f1, epoch)

    def _save(self, model, tokenizer) -> None:
        # Create output folder if it does not exist.
        if not os.path.exists(self._output_folder):
            os.makedirs(self._output_folder)

        # Save a trained model, configuration and tokenizer using
        # 'save_pretrained()'. Then, it is possible to reload the model
        # using 'from_pretrained()' method.
        tokenizer.save_pretrained(self._output_folder)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(self._output_folder)

        # Save metrics
        metrics_path = os.path.join(self._output_folder, 'metrics.csv')
        pd.DataFrame(self._training_stats).to_csv(
            os.path.join(metrics_path), index=False)

        # Save settings
        metrics_path = os.path.join(self._output_folder, 'settings.yaml')
        with open(metrics_path, 'w') as outfile:
            yaml.dump(self._settings, outfile, default_flow_style=False)
