import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class LitCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def train_dataloader(self):
        cfg = self.cfg
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(
            root=cfg.dataset.image_dir,
            train=True,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        cfg = self.cfg
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=cfg.dataset.image_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
