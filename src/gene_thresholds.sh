#!/bin/sh


for genelist in ./gene_list/THOLD_1.0.txt ./gene_list/THOLD_0.05.txt ./gene_list/THOLD_0.01.txt ./gene_list/THOLD_0.005.txt ./gene_list/THOLD_0.001.txt ./gene_list/THOLD_0.0005.txt ./gene_list/THOLD_0.0001.txt  ./gene_list/THOLD_5e-05.txt ./gene_list/THOLD_1e-05.txt ./gene_list/THOLD_5e-06.txt ./gene_list/THOLD_1e-06.txt ./gene_list/THOLD_5e-07.txt ./gene_list/THOLD_1e-07.txt
do
    python ./src/train_IEA.py --genelist $genelist
done