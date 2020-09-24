#!/bin/bash

CUDA_VISIBLE_DEVICES=2

for ((i=1;i<=3;i++))
do
    seed=$RANDOM
    python train_iemocap_hgt.py --base-model 'LSTM' --graph-model --nodal-attention --dropout 0.4 --lr 0.0003 --batch-size 32 --class-weight --l2 0.0 --seed $seed --no-cuda
done
