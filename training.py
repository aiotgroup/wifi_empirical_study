import os
import logging

from torch.utils.data.dataloader import DataLoader

from pipeline import Trainer
from config import TrainConfig

import init_util

logger = logging.getLogger(__name__)


def train(config: TrainConfig):
    if config.gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device

    train_dataset, eval_dataset = init_util.init_dataset(config.dataset_name, config.datasource_path)
    strategy = init_util.init_strategy(config.backbone_name,
                                       config.head_name,
                                       config.strategy_name,
                                       train_dataset.get_n_channels(),
                                       train_dataset.get_seq_lens())

    trainer = Trainer(
        strategy=strategy,
        train_data_loader=DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                     drop_last=True),
        eval_data_loader=DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False),
        num_epoch=config.num_epoch,
        opt_method=config.opt_method,
        lr_rate=config.lr_rate,
        lr_rate_adjust_epoch=config.lr_rate_adjust_epoch,
        lr_rate_adjust_factor=config.lr_rate_adjust_factor,
        weight_decay=config.weight_decay,
        save_epoch=config.save_epoch,
        eval_epoch=config.eval_epoch,
        patience=config.patience,
        check_point_path=config.check_point_path,
        use_gpu=False if config.gpu_device is None else True,
    )

    trainer.training()
