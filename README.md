# PyTorch-Lightning-CIFAR10
"Not too complicated" training code for CIFAR-10 by PyTorch Lightning

This is a refactored repository of [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10). I'm glad if you can use it as a reference.

## Dataset

Details of CIFAR-10 can be found at the following link. [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

## Instalation

I run in the following environment. If you have a similar environment, you can prepare the environment immediately with pipenv.

* Ubuntu 20.04.1 LTS
* CUDA Version 11.0
* Python 3.8.5

```
$ pip install pipenv
$ pipenv sync
```

If you do not have a cuda environment, please use Docker. Build docker with the following command.

```
$ docker-compose up -d dev
```

Run docker with the following command.

```
$ docker run --rm -it --runtime=nvidia \
      -v /mnt/:/mnt \
      -v /home/kuroyanagi/clones/PyTorch-Lightning-CIFAR10/:/work/PyTorch-Lightning-CIFAR10 \
      -u (id -u):(id -g) \
      -e HOSTNAME=(hostname) \
      -e HOME=/home/docker \
      --workdir /work/PyTorch-Lightning-CIFAR10 \
      --ipc host \
      pytorch-lightning-cifar10 bash
```

### Prepare dataset

This repository is implemented in two ways, one is to load CIFAR-10 from **torchvision** and the other is to load CIFAR-10 as a **custom dataset**. I want you to use it as learning how to use custom dataset.

If you want to load CIFAR-10 from **torchvision**, specify config as follows.

```
dataset:
  loading: 'torchvision'
```

If you want to load CIFAR-10 as a **custom dataset**, download the raw image as shown below.

```
$ cd data/
$ bash download_cifar10.sh # Downloads the CIFAR-10 dataset (~161 MB)
```
Also, specify config as custom for loading.

```
dataset:
  loading: 'custom'
```

### Modeling

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

### Train

`train.py` performs training/validation according to the specified config. A checkpoint for each epoch is saved and evaluated for validation.

To execute the experiment of `configs/experiments/train_exp01.yaml`, execute as follows. Specify the output destination as `hydra.run.dir=outputs/train/exp01`.

```
$ pipenv run python train.py +experiments=train_exp01 hydra.run.dir=outputs/train/exp01
```

If you use Docker, execute the following command.

```
$ export TORCH_HOME=/home/docker
$ python train.py +experiments=train_exp01 hydra.run.dir=outputs/train/exp01
```

### Test

Specify `evaluate: True` in config as shown below.

```
runs:
  evaluate: True
```
You can run test with the same code as train.

```
$ pipenv run python train.py +experiments=test_exp01 hydra.run.dir=outputs/test/exp01
```

### Results

The results of TensorBoard are as follows.

![tensorboard](results/tensorboard.png)

### References

* [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)

### TODOs

This repository is still work in progress. Please use with caution.

- [x] check code format with black, isort, vulture.
- [x] Docker and pipenv.
- [] Integration of hydra color logger and PyTorch Lighting logger (Probably not possible).
- [] GPU usage for custom dataset and light weight model(resnet18, MobileNetV3) does not remain high.
- [] In ``evaluate: True``, Accuracy is a strange value.
- [] Fine tuning by torchvision's pretrained model
