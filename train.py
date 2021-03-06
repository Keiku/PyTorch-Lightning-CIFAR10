import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from datamodule import LitCIFAR10DataModule
from model import LitCIFAR10Model


@hydra.main(config_path="./configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:

    if "experiments" in cfg.keys():
        cfg = OmegaConf.merge(cfg, cfg.experiments)

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.runs.gpu_id

    if cfg.runs.logger == "wandb":
        logger = WandbLogger(name=cfg.model.classifier, project=cfg.model.project)
    elif cfg.runs.logger == "tensorboard":
        logger = TensorBoardLogger(
            cfg.train.tensorboard_dir,
            name=cfg.model.classifier,
            version=cfg.model.version,
        )

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if cfg.train.resume and Path(cfg.train.checkpoint).exists():
        resume_checkpoint = cfg.train.checkpoint
    else:
        resume_checkpoint = None

    trainer = Trainer(
        fast_dev_run=cfg.runs.dev,
        logger=logger if not (cfg.runs.dev or cfg.runs.evaluate) else None,
        gpus=-1,
        deterministic=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        max_epochs=cfg.train.num_epochs,
        precision=cfg.runs.precision,
        resume_from_checkpoint=resume_checkpoint,
        callbacks=[checkpoint, lr_monitor],
    )

    datamodule = LitCIFAR10DataModule(cfg)
    model = LitCIFAR10Model(cfg, trainer)

    if cfg.runs.evaluate:
        hparams = OmegaConf.load(cfg.test.hparams)
        model = LitCIFAR10Model.load_from_checkpoint(
            checkpoint_path=cfg.test.checkpoint, **hparams
        )
        trainer.test(model, datamodule.test_dataloader())
    else:
        trainer.fit(model, datamodule)
        # NOTE After changing to pytorch-lightning 1.5.2, omitting the argument of trainer.test() does not work. ?? Discussion #10747 ?? PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10747
        trainer.test(model, datamodule.test_dataloader())


if __name__ == "__main__":
    main()
