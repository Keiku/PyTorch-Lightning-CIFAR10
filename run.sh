#!/bin/bash
docker rm -f $(docker ps -q -a)
docker run --rm --runtime=nvidia \
       -v /mnt/:/mnt \
       -v /mnt/nfs/kuroyanagi/clones/PyTorch-Lightning-CIFAR10/:/work/PyTorch-Lightning-CIFAR10 \
       -v /home/anasys/datasets/:/work/PyTorch-Lightning-CIFAR10/data \
       -u $(id -u):$(id -g) \
       -e HOSTNAME=$(hostname) \
       -e HOME=/home/docker \
       --workdir /work/PyTorch-Lightning-CIFAR10 \
       --ipc host \
       keiku/pytorch-lightning-cifar10 'python train.py +experiments=train_resume_exp01 hydra.run.dir=outputs/train_resume_exp01'
