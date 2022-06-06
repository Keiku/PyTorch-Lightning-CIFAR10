# PyTorch-Lightning-CIFAR10
"Not too complicated" training code for CIFAR-10 by PyTorch Lightning

## :gift: Dataset

Details of CIFAR-10 can be found at the following link. [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

## :package: PyTorch Environment

I am using the following PyTorch environment. See `Pipfile` for more information.

* torch==1.10.0
* torchvision==0.11.1
* pytorch-lightning==1.5.2

## :whale: Instalation

I run in the following environment. If you have a similar environment, you can prepare the environment immediately with pipenv.

* Ubuntu 20.04.1 LTS
* CUDA Version 11.1
* Python 3.8.5

```
pip install pipenv
pipenv sync
```

If you do not have a cuda environment, please use Docker. Build docker with the following command.

```
docker-compose up -d dev
```

Run docker with the following command. The following command is for fish shell. For bash, replace `()` with `$()`.

```
docker run --rm -it --runtime=nvidia \
    -v /mnt/:/mnt \
    -v /mnt/nfs/kuroyanagi/clones/PyTorch-Lightning-CIFAR10/:/work/PyTorch-Lightning-CIFAR10 \
    -u (id -u):(id -g) \
    -e HOSTNAME=(hostname) \
    -e HOME=/home/docker \
    --workdir /work/PyTorch-Lightning-CIFAR10 \
    --ipc host \
    keiku/pytorch-lightning-cifar10 bash
```

## :gift: Prepare dataset

This repository is implemented in two ways, one is to load CIFAR-10 from **torchvision** and the other is to load CIFAR-10 as a **custom dataset**. I want you to use it as learning how to use custom dataset.

If you want to load CIFAR-10 from **torchvision**, specify config as follows.

```
dataset:
  loading: 'torchvision'
```

If you want to load CIFAR-10 as a **custom dataset**, download the raw image as shown below.

```
cd data/
bash download_cifar10.sh # Downloads the CIFAR-10 dataset (~161 MB)
```
Also, specify config as custom for loading.

```
dataset:
  loading: 'custom'
```

## :writing_hand: Modeling

The following three methods are available for modeling.

* **Scratch implementation** resnet18, resnet32, resnet50
* **timm**

When using the scratch implementation of resnet, specify config as follows.

```
model:
  classifier: 'resnet18'
  implementation: 'scratch'

transform:
  normalization: 'cifar10'
```

When using timm's imagenet pretrained model, specify config as follows.

```
model:
  classifier: 'resnet18'
  implementation: 'timm'
  pretrained: True

transform:
  normalization: 'imagenet'
```

## :runner: Train

`train.py` performs training/validation according to the specified config. The checkpoint is saved in the best epoch that monitors the accuracy of validation.

To execute the experiment of `configs/experiments/train_exp01.yaml`, execute as follows. Specify the output destination as `hydra.run.dir=outputs/train_exp01`.

```
pipenv run python train.py +experiments=train_exp01 hydra.run.dir=outputs/train_exp01
```

If you use Docker, execute the following command.

```
export TORCH_HOME=/home/docker
python train.py +experiments=train_exp01 hydra.run.dir=outputs/train_exp01
```

## :running_man: Resume Training

If you want to resume training, specify the following config.

```
train:
  resume: True
  checkpoint: "/mnt/nfs/kuroyanagi/clones/PyTorch-Lightning-CIFAR10/outputs/train_resume_exp01/logs/resnet18/exp01/checkpoints/last.ckpt"
```

Even if you interrupt while using AWS spot instance, you can read `last.ckpt` and restart from the next epoch learning. You can use `run.sh` as a command when restarting.

## :standing_person: Test

Specify `evaluate: True` in config as shown below.

```
runs:
  evaluate: True
```
You can run test with the same code as train.

```
pipenv run python train.py +experiments=test_exp01 hydra.run.dir=outputs/test_exp01
```

The following results are obtained.

```
Global seed set to 0
GPU available: True, used: True
TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 100%|████████████████████████████████████████| 19/19 [00:03<00:00,  5.88it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'acc/test': 93.1743392944336}
--------------------------------------------------------------------------------
```

## :chart_with_upwards_trend: Results

The results of TensorBoard are as follows.

![tensorboard](results/tensorboard.png)

## :zap: PyTorch Lightning API

#### LightningDataModule API in `datamodule.py`

- [x] `LightningDataModule`
    - [ ] `prepare_data()`
    - [ ] `setup()`
    - [x] `train_dataloader()`
    - [x] `val_dataloader()`
    - [x] `test_dataloader()`

#### LightningModule API in `model.py`

- [x] `LightningModule`
    - [x] `forward()`
    - [x] `training_step()`
    - [x] `validation_step()`
    - [x] `test_step()`
    - [x] `configure_optimizers()`

#### Metrics in `model.py`

- [x] `torchmetrics.Accuracy()`

#### API in `train.py`

#### Lightning CLI API

- [ ] `LightningCLI()`

#### Trainer API

- [x] `Trainer`
    - [x] `.fit()`
    - [x] `ModelCheckpoint()`
    - [x] `LearningRateMonitor()`
    - [x] `.load_from_checkpoint()`
    - [x] `.test()`

#### Loggers API

- [x] `TensorBoardLogger()`
- [x] `WandbLogger()`


## :closed_book: References

* [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)

## :rocket: TODOs

- [x] Check code format with black, isort, vulture
- [x] Docker and pipenv
- [x] TensorBoard and Wandb logging
- [x] Loading by custom dataset
- [x] Transfer learning by timm
- [x] Simple evaluation using `.load_from_checkpoint()`
- [x] Resume training
- [x] Use torchmetrics
- [ ] Transform by albumentations
- [ ] Remove Hydra and use Lightning CLI and config files
- [ ] Add Fault-tolerant Training
- [ ] Add EarlyStopping
- [ ] Remove pipenv and add requirements.txt or poetry
