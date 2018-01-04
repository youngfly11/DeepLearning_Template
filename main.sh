#!/usr/bin/env bash
nohup python main.py --lr 1e-3 --epoch 150 --batch-size 512 --weight-decay 1e-5 --env glimpse_8 --print-freq 1 --Glimpse 6 | tee glimpse_8.out