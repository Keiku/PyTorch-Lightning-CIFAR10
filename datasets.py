import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train, transform=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.cfg = cfg
        self.split_dir = "train" if train else "test"
        self.root_dir = Path(cfg.dataset.root_dir)
        self.image_dir = self.root_dir / "cifar" / self.split_dir
        self.file_list = [p.name for p in self.image_dir.rglob("*") if p.is_file()]
        self.labels = [re.split("_|\.", l)[1] for l in self.file_list]
        self.targets = self.label_mapping(cfg)

    def label_mapping(self, cfg):
        labels = self.labels
        label_mapping_path = Path(cfg.dataset.root_dir) / "cifar/labels.txt"
        df_label_mapping = pd.read_table(label_mapping_path.as_posix(), names=["label"])
        df_label_mapping["target"] = range(cfg.train.num_classes)

        label_mapping_dict = dict(
            zip(
                df_label_mapping["label"].values.tolist(),
                df_label_mapping["target"].values.tolist(),
            )
        )

        targets = [label_mapping_dict[i] for i in labels]
        return targets

    def __getitem__(self, index):
        filename = self.file_list[index]
        targets = self.targets[index]
        image_path = self.image_dir / filename
        image = Image.open(image_path.as_posix())

        if self.transform is not None:
            transform = self.transform
            image = transform(image)

        return image, targets

    def __len__(self):
        return len(self.file_list)
