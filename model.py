import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

from models.resnet import resnet18, resnet34, resnet50
from schduler import WarmupCosineLR

all_classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


class LitCIFAR10Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.cfg.train.classifier]

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

    def configure_optimizers(self):
        cfg = self.cfg
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = cfg.train.num_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
