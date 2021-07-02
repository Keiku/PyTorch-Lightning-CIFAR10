import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
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
        logger = WandbLogger(name=cfg.model.classifier, project="cifar10")
    elif cfg.runs.logger == "tensorboard":
        logger = TensorBoardLogger(cfg.train.tensorboard_dir, name=cfg.model.classifier)

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=True)

    trainer = Trainer(
        fast_dev_run=cfg.runs.dev,
        logger=logger if not (cfg.runs.dev or cfg.runs.evaluate) else None,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=cfg.train.num_epochs,
        checkpoint_callback=checkpoint,
        precision=cfg.runs.precision,
        resume_from_checkpoint=cfg.train.checkpoint,
    )

    datamodule = LitCIFAR10DataModule(cfg)
    model = LitCIFAR10Model(cfg)

    if cfg.runs.evaluate:
        trainer.test(model, datamodule.test_dataloader())
    else:
        trainer.fit(model, datamodule)
        trainer.test()


if __name__ == "__main__":
    main()
