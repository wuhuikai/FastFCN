#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset ade20k \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_ade20k_train

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset ade20k \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_ade20k_trainval \
    --train-split trainval --lr 0.001 --epochs 20 --ft --resume {MODEL_PATH}

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split test --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split test --mode test --ms
