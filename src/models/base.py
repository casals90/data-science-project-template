import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models import utils as model_utils
from src.tools import utils as tools_utils
from src.tools.startup import logger

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class TransformerBase:
    def __init__(
            self, _settings, pipeline_name: str, args_key: str) -> None:
        self._settings = _settings
        pipeline_settings = _settings[pipeline_name][args_key]

        self._model = None

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        logger.info(f'Device: {self._device}')

        self._execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f'Initializing execution {self._execution_id}')

        self._args = pipeline_settings
        self._args['writer']['log_dir'] = \
            self._args['writer']['log_dir'].format(
                execution_id=self._execution_id)

        self._writer = SummaryWriter(**self._args['writer'])

        self._output_folder = _settings[pipeline_name]['load']['path'].format(
            execution_id=self._execution_id)

    @staticmethod
    def _create_dataloader(dataset, sampler_settings, dataloader_args):
        sampler_params = sampler_settings['params'].copy() if isinstance(
            sampler_settings['params'], dict) else {}
        sampler_params['data_source'] = dataset
        sampler = tools_utils.import_library(
            sampler_settings['module'], sampler_params)

        dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)

        return dataloader

    def _model_step(self, model, batch) -> tuple:
        return model_utils.model_step(model, batch, self._device)
