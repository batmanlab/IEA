#!/bin/sh


for L in 1 2 3 4 5
do
    python ./src/train_IEA.py --L $L
done