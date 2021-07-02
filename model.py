import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy

from models.resnet import resnet18, resnet34, resnet50
from torchvision.models import resnet18
import timm
from schduler import WarmupCosineLR

classifiers = {
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

        self.model = classifiers[self.cfg.model.classifier]

    def get_model(cfg):
        if cfg.model.implementation == 'scratch':
            model = all_classifiers[self.cfg.model.classifier]
        elif cfg.model.implementation == 'torchvision':
            model = resnet18(pretrained=cfg.model.pretrained)
        elif cfg.model.implementation == 'timm':
            model = timm.create_model(
                cfg.model.classifier,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.train.num_classes)
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
