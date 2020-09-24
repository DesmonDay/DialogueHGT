#!/bin/bash

CUDA_VISIBLE_DEVICES=1

for ((i=1;i<=3;i++))
do
    seed=$RANDOM
    CUDA_VISIBLE_DEVICES=1 python train_iemocap_hgt.py --base-model 'LSTM' --graph-model --nodal-attention --dropout 0.4 --lr 0.0003 --batch-size 32 --class-weight --l2 0.0 --seed $seed
done
