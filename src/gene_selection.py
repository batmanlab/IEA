import sys
import numpy as np
import os

import torch
import torch.distributed as dist


from importlib import reload
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import  RidgeCV

import utility

from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser(description='Selecting genes with the image features. The selected genes are saved in the output files.')
parser.add_argument('--output_dir', type=str, default="./gene_list/", help='The selected genes are saved in the output directory.')

args = parser.parse_args()


# Create the output directory.
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    

#Load the data
image_gene_dataset = utility.image_gene_dataset(train = True, cv = None,)
sid, image_features, genes = image_gene_dataset[:]

#Conduct PCA for the image features.
X_raw = image_features.view(-1, 581 * 128).cpu().data.numpy()
M_PCA = PCA(n_components = 128)
X = M_PCA.fit_transform(X_raw)

genes = genes.numpy()

p_values = []

#Compute the p-values.
for gene_idx in range( genes.shape[1] ):
    y = genes[:, gene_idx]
    M_OLS = sm.OLS(y, X).fit()
    p_values.append(M_OLS.f_pvalue)
    
    if gene_idx % 100 == 0:
        print( "Processing {} / {}".format(gene_idx, genes.shape[1] ))

#Adjust the p-values with Benjamini/Hochberg method.
_, adj_p, _, _ = multipletests(p_values, method = "fdr_bh")


# Get list of genes with different thresholds
p_thresholds = [1., .05, .01, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7 ] 

for ppp in p_thresholds:
    
    filename = os.path.join(args.output_dir, "THOLD_{}.txt".format(ppp))
    
    f = open(filename, "w")
    f.write("index, gene name\n")
    for iii in np.where( adj_p <= ppp )[0]:
        f.write("{},{}\n".format(iii, image_gene_dataset.genelist[iii]))
    f.close()
    
    
    