# DialogueHGT++

To reproduce the IEMOCAP dataset results, run:
```
python train_iemocap_hgt.py --base-model 'LSTM' --nodal-attention --dropout 0.4 --lr 0.0003 --batch-size 32 --class-weight --l2 0.0 --no-cuda
```
