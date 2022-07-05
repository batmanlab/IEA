import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import r2_score
from torch.nn.parameter import Parameter
from sklearn.linear_model import Lasso
import multiprocessing
from torch.utils.data import DataLoader
from functools import partial
from sklearn.linear_model import Ridge
from monai.networks.nets import FullyConnectedNet

from scipy.special import expit


class IEA(nn.Module):    
    def __init__(self, loader = None, eval_loader = None, idx_gene = np.arange(18487), 
                 L = 16, tau = 10., dropout = 0,
                 alpha = 1., beta = 0.1, gamma = 0., rho = 0., trd = 1e-5, 
                 lr = 1e-4, lr2 = 1e-2, lr_D = 1e-3, 
                 n_rep = 5, filename = None, n_patch = 64, seed = 0):
        """
        Model initialization.
        """
        
        super(IEA, self).__init__()        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.dropout = dropout
        
        self.lr = lr
        self.lr2 = lr2
        self.lr_D = lr_D
        self.n_rep = n_rep
        self.idx_gene = idx_gene
        self.n_patch = n_patch
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.L = L
        self.d_out = len(self.idx_gene)
        
        self.encoder = FullyConnectedNet(128, L * 2, hidden_channels = (96, 64, 48), bias = True, dropout = self.dropout)
        self.linear = nn.Linear(self.L, self.d_out)
        
        
        self.optim = optim.Adam( self.encoder.parameters(), 
            lr=self.lr, betas=(0.0,0.999), eps=1e-8)
        self.optim2 = optim.Adam(   self.linear.parameters(),
            lr=self.lr2, betas=(0.0,0.999), eps=1e-8)
        
        
        self.loader = loader
        self.eval_loader = eval_loader
        
        if filename is None:
            self.filename = \
                    "models/model_L{}_beta{}_gamma{}_genes{}".format(
                    L, beta, gamma, len(self.idx_gene))
        else:
            self.filename = filename
        

        self.r2_opt = -1e30
        
    def weights_init_uniform(self):
        """
        Initialize the model with kaiming uniform
        """
        classname = self.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            nn.init.kaiming_uniform_(self.weight.data,  a=np.sqrt(5))        

    def PoG(self, mu_np, log_var_np): 
        """
        Given the means and variances of Guasian distributions, return mean and variances of the product of the distributions. 
        """
        var_np = torch.exp(log_var_np)
        var_n =  1. /  ( 1. / var_np) .sum(1) 
        mu_n = var_n * ( 1. / var_np * mu_np).sum(1)
        log_var_n = torch.log(var_n)
        
        return mu_n, log_var_n
        
        
    def reparameterize(self, mu, log_var):
        """
        Re-parameterization tricks for a Guassian distribution. The mean and the logarithm of variance are provided.
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
                
        
    def KL_Divergence(self, mu, log_var):
        """
        Computing the KL divergence of a Gaussian distribution to the standard Gaussian distribution with zero mean and unit variance.
        """
        return .5 * ( -1 - log_var + log_var.exp() + mu.pow(2)  )
    
    def forward(self, X_npd, n_patch = None, non_prb = False):
        n_sample = X_npd.shape[0]
        
        if n_patch is None:
            X_pd = X_npd.view(n_sample * 581, 128)
            n_patch = 581
            
        else:
            p_idx = np.random.choice( 581, size = n_patch * n_sample, replace = True)
            n_idx = np.array( [ [iii] * n_patch for iii in range(n_sample) ], int).flatten()
            X_pd = X_npd[n_idx, p_idx, :]

            
        res_pl = self.encoder(X_pd)
        mu_pl, log_var_pl = res_pl[:, :self.L], res_pl[:, self.L:]
        mu_npl = mu_pl.view(n_sample, n_patch, self.L)
        log_var_npl = log_var_pl.view(n_sample, n_patch, self.L)
        mu_nl, log_var_nl = self.PoG(mu_npl, log_var_npl)
        
        if non_prb:
            Z_nl = mu_nl
        else:
            Z_nl = self.reparameterize(mu_nl, log_var_nl)
        y_nm = self.linear(Z_nl)
        
        return y_nm, mu_npl, log_var_npl, Z_nl
    
    def RBF(self, X):
        """
        Compute the RBF for the input matrix X
        """
        N, D = X.shape
        size = (N, N, D)
        
        X1 = X.unsqueeze(1).expand(size).cuda()
        X2 = X.unsqueeze(0).expand(size).cuda()

        dis2 = ( (X1 - X2) ** 2 ).sum(2)
        sigma2 = torch.median(dis2[dis2 >0])

        K = torch.exp( - dis2 / sigma2 / 2.)
        
        return K
        
        
    def HSIC(self, Z1, Z2):
        
        """
        Return HSIC value between Z1 and Z2.
        """
        
        N, _ = Z1.shape
        K1 = self.RBF(Z1) + 1e-3 * torch.eye(N).cuda()
        K2 = self.RBF(Z2) + 1e-3 * torch.eye(N).cuda()
        A = torch.linalg.cholesky(K1.cuda())
        B = torch.linalg.cholesky(K2.cuda())

        H = ( torch.eye(N) - 1. / N * torch.ones(N, N) ).cuda()

        HSIC = 1. / (N - 1) ** 2 * (( torch.matmul( (torch.matmul(H , A)).T , B )  ) ** 2).sum()
        return HSIC
       
        
    def train_model(self, n_step = 10000, n_init_step = 1, n_repeat = 1):
        """
        Model training. n_step is the number of steps of training. n_init_step is the number of steps for initialization. n_repeat is the number of re-initialization for the training.
        """

        self.train()
        f = open(self.filename.replace("models/", "logs/") + ".txt", "a", buffering=1)
        
        for rep in range(n_repeat):
            self.step = 0
            self.weights_init_uniform()
        
            while True:
                data_iter = iter(self.loader)

                for sid, rep, gene in data_iter:

                    self.train()

                    """
                    Updating parameters
                    """


                    self.optim.zero_grad()
                    self.optim2.zero_grad()

                    X_npd = rep.cuda().view(-1, 581, 128)
                    y_nm = gene.cuda()[:, self.idx_gene]
                    batch_size = X_npd.shape[0]

                    y_pred_nm, mu_npl, log_var_npl, _ = self.forward(X_npd, n_patch = self.n_patch)
                    y_pred_nm_, _, _, rep_nl = self.forward(X_npd, non_prb = True, n_patch = self.n_patch)

                    KLD = self.KL_Divergence(mu_npl, log_var_npl).mean()

                    HSIC_loss = torch.tensor(0.).cuda()
                    for iii in range(self.L):
                        for jjj in range(iii + 1, self.L):                    
                            HSIC = self.HSIC(rep_nl[:, [iii]], rep_nl[:, [jjj]])
    #                         print(iii, jjj, HSIC)
                            HSIC_loss += HSIC





                    MSE = ( (y_nm - y_pred_nm ) ** 2 ).mean()
                    MSE_ = ( (y_nm - y_pred_nm_ ) ** 2 ).mean()


                    if self.step > n_init_step:
                        gamma = self.gamma
                    else:
                        gamma = self.gamma  * expit( ( self.step - ( n_init_step /2 ) ) / (n_init_step * 20 + 1e-5) )

                    loss = MSE + self.alpha * MSE_ + gamma * HSIC_loss + self.beta * KLD



                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1e-3)

                    self.step += 1


                    self.optim.step()
                    self.optim2.step()


                    """
                    Evaluating
                    """

                    if self.step % self.n_rep == 0:

                        txt = "Step: {}\n".format(self.step)
                        txt += "MSE: {}\n".format( MSE.item() )
                        txt += "HSIC Loss: {}\n".format( HSIC_loss.item() )
                        txt += "Loss: {}\n\n".format(loss.item())

                        print(txt)                    
                        f.write(txt)


                        r2 = self.eval_model()

                        txt = "R2:{}\n".format(r2.item())
                        txt += "opt R2:{}\n\n".format(self.r2_opt)

                        print(txt)

                        if r2 > self.r2_opt and self.step > n_init_step:
                            self.r2_opt = r2.item()
                            torch.save( self.state_dict(), self.filename )

                        f.write(txt)    



                    if self.step > n_step:
                        break

                if self.step > n_step:
                    break
                    
        f.close()

    def eval_model(self):
        """
        Model evaluation. Return the r2 with the test set.
        """
        
        self.eval()
        
        data_iter = iter(self.eval_loader)
        sid, rep, gene = next(data_iter)
        X = rep.cuda().view(-1, 581, 128)
        y = gene[:, self.idx_gene].cpu().data.numpy()
        
        with torch.no_grad():
            y_pred_nm, _, _, _ = self.forward(X, non_prb = True)
        y_predicted = y_pred_nm.cpu().data.numpy()
        
        r2 = r2_score(y, y_predicted)
        
        self.train()
        
        return r2
        
        
        
    