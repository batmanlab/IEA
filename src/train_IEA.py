import sys
import torch
import numpy as np
import pandas as pd


from torch.utils.data import DataLoader
import utility
import IEA_model


L = int(sys.argv[1])
thold = float(sys.argv[2])

gamma = float(sys.argv[3])
beta = float(sys.argv[4])

for cv in range(5):

    df = pd.read_csv("gene_list_PCA/THOLD_{}.txt".format(thold))
    idx_gene = df["index"].to_numpy()

    # idx_gene = np.load("../image_gene/ICA_no_correction/gene_idx_CV_major0.001.npy")

    data_train = utility.image_gene_dataset(train = True, cv = cv, n_cv = 5)
    data_validate = utility.image_gene_dataset(train = False, cv = cv, n_cv = 5)

    train_loader = DataLoader(data_train, batch_size = 32)
    validate_loader = DataLoader(data_validate, batch_size = 500)


    filename = "models/model_L{}_cv{}_beta{}_gamma{}_genes{}".format(
               L, cv, beta, gamma, len(idx_gene))

    M = IEA_model.IEA(train_loader, validate_loader, 
                      idx_gene = idx_gene, gamma = gamma, beta = beta, L = L, 
                      n_patch = 32, n_rep = 10, filename = filename).cuda()


    M.train_model(8000, 5000, 1)


