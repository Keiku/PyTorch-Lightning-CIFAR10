import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from datasets import CIFAR10Dataset


class LitCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mean = self.set_normalization(cfg)['mean']
        self.std = self.set_normalization(cfg)['std']

    def set_normalization(self, cfg):
        # Image classification on the CIFAR10 dataset - Albumentations Documentation https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if cfg.transform.normalization == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif cfg.transform.normalization == 'imagenet':
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std =(0.229, 0.224, 0.225)
        return {'mean': mean, 'std': std}

    def get_dataset(self, cfg, train, transform):
        if cfg.dataset.loading == 'torchvision':
            dataset = CIFAR10(
                root=cfg.dataset.root_dir,
                train=train,
                transform=transform,
                download=train,
            )
        elif cfg.dataset.loading == 'custom':
            dataset = CIFAR10Dataset(
                cfg=cfg,
                train=train,
                transform=transform,
            )
        else:
            raise NotImplementedError
        return dataset

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
        dataset = self.get_dataset(
            cfg=cfg,
            train=True,
            transform=transform,
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
        dataset = self.get_dataset(
            cfg=cfg,
            train=False,
            transform=transform
        )
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
