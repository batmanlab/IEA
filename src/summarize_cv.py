import sys
sys.path.append("./src")

import IEA_model
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import utility

from sklearn.linear_model import LinearRegression

import argparse
import os

parser = argparse.ArgumentParser(description='Summarize the IEAs generated in cross validations to generate the final IEAs. The script will generate three files: IEAs for the training set (IEA_train.csv), IEAs for the test set (IEA_test.csv) and IEAs for the Phase-1 data ("IEA_P1.csv").')
parser.add_argument('--output_dir', type=str, default="./output/", help='The output directory for the final IEAs')
parser.add_argument('--model_list', type=str, default=None, help='The file that contains the list of trained models with cross-validation')
parser.add_argument('--genelist', type=str, default="./gene_list/THOLD_0.01.txt", help='The filename for the gene list that is used for training.')
parser.add_argument('--L', type=int, default=2, help='Number of axes')

args = parser.parse_args()


# Loading the data.
data_train = utility.image_gene_dataset(train = True, cv = None)
data_test = utility.image_gene_dataset(train = False, cv = None)


sid_train, features_train, genes_train = data_train[:]
sid_test, features_test, genes_test = data_test[:]

sid_P1, features_P1 = utility.load_SSL_features_P1()
features_P1 = torch.tensor(features_P1)

df_gene = pd.read_csv(args.genelist)
idx_gene = df_gene["index"].to_numpy()


if args.model_list is None:
    beta = .1
    gamma = 1.0
    L = args.L
    n_genes = len(idx_gene)
    model_list = ["./models/model_L{}_cv{}_beta{}_gamma{}_genes{}".format(
               L, cv, beta, gamma, n_genes) for cv in range(5) ]    
else:
    with open(args.model_list) as f:
        model_list = [line.rstrip() for line in f]


IEA_train_cv = []
IEA_test_cv = []
IEA_P1_cv = []


# Load the cross-validation models and put the results in lists.

M = IEA_model.IEA( L = L,  idx_gene = idx_gene ).cuda()

n_cv = len(model_list)

for cv in range(n_cv):
    filename = model_list[cv]

    M.load_state_dict( torch.load(filename) )
    
    with torch.no_grad():
        _, _, _, IEA_train = M.forward( features_train.cuda(), non_prb = True )
        IEA_train_cv.append( IEA_train.cpu().data.numpy() )
        
        _, _, _, IEA_test = M.forward( features_test.cuda(), non_prb = True  )
        IEA_test_cv.append( IEA_test.cpu().data.numpy() )

        _, _, _, IEA_P1 = M.forward( features_P1.cuda(), non_prb = True  )
        IEA_P1_cv.append( IEA_P1.cpu().data.numpy() )
        
IEA_train_cv = np.array( IEA_train_cv )
IEA_test_cv = np.array( IEA_test_cv )
IEA_P1_cv = np.array( IEA_P1_cv )
    

# Align the IEAs from different cross validation. Normalize each IEA such that it has zero-mean and unit-variance.
IEA_train_aligned = []
IEA_test_aligned = []
IEA_P1_aligned = []

for cv in range(0,5):
    IEA_cat = np.concatenate( [ IEA_train_cv[ 0, :, : ], IEA_train_cv[ cv, :, : ] ], axis = 1 )
    corrcoef = np.corrcoef(IEA_cat.T)
    
    IEA_train = []
    IEA_test = []
    IEA_P1 = []
    for lll in range(L):
        
        
        corr_lll = corrcoef[lll, L:]
        idx_lll = np.argmax( np.abs(corr_lll) )
        
        
        mean = IEA_train_cv[cv, :, idx_lll].mean()
        std = IEA_train_cv[cv, :, idx_lll].std()
        
        if corr_lll[idx_lll]>0:
            IEA_train.append( ( IEA_train_cv[cv, :, idx_lll] - mean ) / std )
            IEA_test.append( ( IEA_test_cv[cv, :, idx_lll] - mean ) / std  )
            IEA_P1.append(( IEA_P1_cv[cv, :, idx_lll] - mean ) / std  )
        else:
            IEA_train.append( - ( IEA_train_cv[cv, :, idx_lll] - mean) / std )
            IEA_test.append( - ( IEA_test_cv[cv, :, idx_lll] - mean) / std )
            IEA_P1.append( - ( IEA_P1_cv[cv, :, idx_lll] - mean ) / std  )
        
    
    IEA_train_aligned.append( np.array(IEA_train).T )
    IEA_test_aligned.append( np.array(IEA_test).T )
    IEA_P1_aligned.append( np.array(IEA_P1).T )
    
IEA_train_aligned = np.array(IEA_train_aligned)
IEA_test_aligned = np.array(IEA_test_aligned)
IEA_P1_aligned = np.array(IEA_P1_aligned)
    
    
# Final IEAs are given by the mean values of the aligned IEA.    
IEA_train_final = IEA_train_aligned.mean(0)
IEA_test_final = IEA_test_aligned.mean(0)
IEA_P1_final = IEA_P1_aligned.mean(0)    

# Convert the results into DataFrame.
df_train = pd.DataFrame(data = IEA_train_final, columns = ["IEA{}".format(iii) for iii in range(L) ], index = sid_train)
df_test = pd.DataFrame(data = IEA_test_final, columns = ["IEA{}".format(iii) for iii in range(L) ], index = sid_test)
df_P1 = pd.DataFrame(data = IEA_P1_final, columns = ["IEA{}".format(iii) for iii in range(L) ], index = sid_P1)

# Save the results in the output directory
df_train.to_csv( os.path.join(args.output_dir, "IEA_train.csv"), index_label = "sid")
df_test.to_csv(os.path.join(args.output_dir, "IEA_test.csv"), index_label = "sid")
df_P1.to_csv(os.path.join(args.output_dir, "IEA_P1.csv"), index_label = "sid")

