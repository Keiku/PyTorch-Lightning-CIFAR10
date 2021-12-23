import pytorch_lightning as pl
import timm
import torch
import torchmetrics

from models.resnet import resnet18, resnet34, resnet50
from schduler import WarmupCosineLR

from pytorch_lightning import LightningModule, Trainer, LightningDataModule


classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


class LitCIFAR10Model(pl.LightningModule):
    def __init__(self, cfg, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.cfg = cfg

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

        self.model = self.get_model(cfg)
        self.trainer = trainer

    def get_model(self, cfg):
        if cfg.model.implementation == "scratch":
            model = classifiers[self.cfg.model.classifier]
        elif cfg.model.implementation == "timm":
            model = timm.create_model(
                cfg.model.classifier,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.train.num_classes,
            )
        else:
            raise NotImplementedError()

        return model

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)
        return loss

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def setup_steps(self, stage=None):
        # NOTE There is a problem that len(train_loader) does not work.
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        cfg = self.cfg
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = cfg.train.num_epochs * self.setup_steps(self)
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
