import sys
sys.path.append("./src")


import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import utility
import IEA_model
import os


import argparse



parser = argparse.ArgumentParser(description='Train the IEA model with cross-validation. The model after training will be save in files')

parser.add_argument('--L', type=int, default=2, help='Number of axes')
parser.add_argument('--gamma', type=float, default=1.0, help='The real-valued hyper-parameter gamma that controls the weight of the HSIC penalty in the objective function')
parser.add_argument('--beta', type=float, default=.1, help='The real-valued hyper-parameter beta that controls the weight of the KL divergence in the objective function')
parser.add_argument('--seed', type=int, default=0, help='The seed for the random number generator.')
parser.add_argument('--n_patch', type=int, default=32, help='The number of randomly selected patches for each subject used during the training.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--n_record', type=int, default=10, help='The number of epochs for logging')
parser.add_argument('--genelist', type=str, default="./gene_list/THOLD_0.01.txt", help='The filename for the gene list that is used for training.')
parser.add_argument('--filelist', type=str, default=None, help='A text file that gives the filename list of for the saved models. The number of files in the list corresponds to the number of cross validations. If this is not provided, a default list is used.' )
parser.add_argument('--n_step', type=int, default=8000, help='Number of training steps.' )
parser.add_argument('--n_init_step', type=int, default=5000, help='Number of initialization steps during training.' )



args = parser.parse_args()


L = args.L


beta = args.beta
gamma = args.gamma
seed = args.seed
n_patch = args.n_patch
n_record = args.n_record
batch_size = args.batch_size
n_step = args.n_step
n_init_step = args.n_init_step


df_gene = pd.read_csv(args.genelist)
idx_gene = df_gene["index"].to_numpy()

if not os.path.exists("../models/"):
    os.makedirs("../models/")


if args.filelist is None:
    filelist = ["./models/model_L{}_cv{}_beta{}_gamma{}_genes{}".format(
               L, cv, beta, gamma, len(idx_gene)) for cv in range(5) ]    
else:
    with open(args.filelist) as f:
        filelist = [line.rstrip() for line in f]
        
n_cv = len(filelist)


for cv in range(n_cv):
    #Load the data
    data_train = utility.image_gene_dataset(train = True, cv = cv, n_cv = n_cv)
    data_validate = utility.image_gene_dataset(train = False, cv = cv, n_cv = n_cv)

    train_loader = DataLoader(data_train, batch_size = args.batch_size)
    validate_loader = DataLoader(data_validate, batch_size = 500)


    filename = filelist[cv]

    #Train the model.
    M = IEA_model.IEA(train_loader, validate_loader, 
                      idx_gene = idx_gene, gamma = gamma, beta = beta, L = L, seed = seed, 
                      n_patch = n_patch, n_record = n_record, filename = filename).cuda()

    M.train_model(n_step, n_init_step)




