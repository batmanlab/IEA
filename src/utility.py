import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
import torch
from sklearn.decomposition import PCA
import os



def load_data():
    """
    Load the pandas dataframe that contains the cross-sectional and longitudinal data.
    """
    
    # List of the variables returned in the primary dataset.
    covariances = ["Age_P1", "gender", "race", 
                   "FEV1pp_utah_P1", "FEV1_FVC_utah_P1", "BMI_P1", "smoking_status_P1",
                   "SGRQ_scoreTotal_P1","MMRCDyspneaScor_P1", 
                   "distwalked_P1", "ATS_PackYears_P1", "Frequent_Exacerbator_P1",  
                   "pctEmph_Thirona_P1", "Perc15_Insp_Thirona_P1", "Pi10_Thirona_P1", "pctGasTrap_Thirona_P1", 
                   "WallAreaPct_seg_Thirona_P1",  
                   "finalGold_P1",
                   "delta_FEV1pp_P1P2", "delta_FEV1FVC_P1P2", "delta_pctEmph_P1P2", "delta_perc15_P1P2",
                   "delta_pctGasTrap_P1P2", "delta_Pi10_P1P2", "delta_WApct_P1P2",
                   "Age_P2",
                   "FEV1pp_utah_P2", "FEV1_FVC_utah_P2", "BMI_P2", "smoking_status_P2",
                   "SGRQ_scoreTotal_P2","MMRCDyspneaScor_P2", 
                   "distwalked_P2", "ATS_PackYears_P2", "Frequent_Exacerbator_P2" ,  
                   "pctEmph_Thirona_P2", "Perc15_Insp_Thirona_P2", "Pi10_Thirona_P2", "pctGasTrap_Thirona_P2", 
                   "WallAreaPct_seg_Thirona_P2", 
                   "delta_FEV1pp_P2P3", "delta_FEV1FVC_P2P3",
                   "Frequent_Exacerbator_P3", 
                   "finalGold_P2"
                   
               ] 
    
    # List of the subtypes returned in the subtype dataset.
    subtypes_used = ["PCA_emphysema_axis", "PCA_airway_axis", "KM_v2"]    

    filename = "/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/P1-P2-P3_10k/COPDGene_P1P2P3_Flat_SM_NS_Mar20.txt"

    res = np.loadtxt(filename, str, delimiter = "\t")
    df = pd.DataFrame(res[1:], columns = res[0]).set_index("sid")
    
    
    #Computing the longitudinal change variables
    
    FEV1pp_P1 = df["FEV1pp_utah_P1"]
    FEV1pp_P1[ FEV1pp_P1 == "" ] = "nan"
    FEV1pp_P1 = FEV1pp_P1.astype(float)

    FEV1pp_P2 = df["FEV1pp_utah_P2"]
    FEV1pp_P2[ FEV1pp_P2 == "" ] = "nan"
    FEV1pp_P2 = FEV1pp_P2.astype(float)

    FEV1pp_P3 = df["FEV1pp_utah_P3"]
    FEV1pp_P3[ FEV1pp_P3 == "" ] = "nan"
    FEV1pp_P3 = FEV1pp_P3.astype(float)

    FEV1FVC_P1 = df["FEV1_FVC_utah_P1"]
    FEV1FVC_P1[ FEV1FVC_P1 == "" ] = "nan"
    FEV1FVC_P1 = FEV1FVC_P1.astype(float)

    FEV1FVC_P2 = df["FEV1_FVC_utah_P2"]
    FEV1FVC_P2[ FEV1FVC_P2 == "" ] = "nan"
    FEV1FVC_P2 = FEV1FVC_P2.astype(float)

    FEV1FVC_P3 = df["FEV1_FVC_utah_P3"]
    FEV1FVC_P3[ FEV1FVC_P3 == "" ] = "nan"
    FEV1FVC_P3 = FEV1FVC_P3.astype(float)

    pct_emph_P1 = df["pctEmph_Thirona_P1"]
    pct_emph_P1[ pct_emph_P1 == "" ] = "nan"
    pct_emph_P1 = pct_emph_P1.astype(float)

    pct_emph_P2 = df["pctEmph_Thirona_P2"]
    pct_emph_P2[ pct_emph_P2 == "" ] = "nan"
    pct_emph_P2 = pct_emph_P2.astype(float)


    perc15_P1 = df["Perc15_Insp_Thirona_P1"]
    perc15_P1[ perc15_P1 == "" ] = "nan"
    perc15_P1 = perc15_P1.astype(float)

    perc15_P2 = df["Perc15_Insp_Thirona_P2"]
    perc15_P2[ perc15_P2 == "" ] = "nan"
    perc15_P2 = perc15_P2.astype(float)

    gastrap_P1 = df["pctGasTrap_Thirona_P1"]
    gastrap_P1[ gastrap_P1 == "" ] = "nan"
    gastrap_P1 = gastrap_P1.astype(float)

    gastrap_P2 = df["pctGasTrap_Thirona_P2"]
    gastrap_P2[ gastrap_P2 == "" ] = "nan"
    gastrap_P2 = gastrap_P2.astype(float)

    Pi10_P1 = df["Pi10_Thirona_P1"]
    Pi10_P1[ Pi10_P1 == "" ] = "nan"
    Pi10_P1 = Pi10_P1.astype(float)

    Pi10_P2 = df["Pi10_Thirona_P2"]
    Pi10_P2[ Pi10_P2 == "" ] = "nan"
    Pi10_P2 = Pi10_P2.astype(float)

    WApct_P1 = df["WallAreaPct_seg_Thirona_P1"]
    WApct_P1[ WApct_P1 == "" ] = "nan"
    WApct_P1 = WApct_P1.astype(float)

    WApct_P2 = df["WallAreaPct_seg_Thirona_P2"]
    WApct_P2[ WApct_P2 == "" ] = "nan"
    WApct_P2 = WApct_P2.astype(float)


    N_years_P2P3 = df["years_CT2_CT3"]
    N_years_P2P3[ N_years_P2P3 == "" ] = "nan"
    N_years_P2P3 = N_years_P2P3.astype(float)

    N_years_P1P3 = df["years_CT1_CT3"]
    N_years_P1P3[ N_years_P1P3 == "" ] = "nan"
    N_years_P1P3 = N_years_P1P3.astype(float)
    
    N_years_P1P2 = N_years_P1P3 - N_years_P2P3  

    
    
    df["delta_FEV1pp_P1P2"] = ( (FEV1pp_P2 - FEV1pp_P1) / N_years_P1P2 ).astype(str)
    df["delta_FEV1FVC_P1P2"] =( (FEV1FVC_P2 - FEV1FVC_P1) / N_years_P1P2 ).astype(str)
    df["delta_pctEmph_P1P2"] = ( (pct_emph_P2 - pct_emph_P1) / N_years_P1P2 ).astype(str)
    df["delta_perc15_P1P2"] =( (perc15_P2 - perc15_P1) / N_years_P1P2 ).astype(str)
    df["delta_pctGasTrap_P1P2"] = ( (gastrap_P2 - gastrap_P1) / N_years_P1P2 ).astype(str)
    df["delta_Pi10_P1P2"] = ( (Pi10_P2 - Pi10_P1) / N_years_P1P2 ).astype(str)
    df["delta_WApct_P1P2"] =( (WApct_P2 - WApct_P1) / N_years_P1P2 ).astype(str)

    df["delta_FEV1pp_P2P3"] = ( (FEV1pp_P3 - FEV1pp_P2) / N_years_P2P3 ).astype(str)
    df["delta_FEV1FVC_P2P3"] =( (FEV1FVC_P3 - FEV1FVC_P2) / N_years_P2P3 ).astype(str)

    
    
    Exacerbation_P1 = df["Exacerbation_Frequency_P1"]
    Exacerbation_P1[Exacerbation_P1 == ""] = "nan"
    Exacerbation_P1 = Exacerbation_P1.astype(float)
    df["Frequent_Exacerbator_P1"] = Exacerbation_P1 >= 2 
    df["Frequent_Exacerbator_P1"] = df["Frequent_Exacerbator_P1"].astype(float).astype(str) 
    df.loc[np.isnan(Exacerbation_P1), "Frequent_Exacerbator_P1"] = "nan"


    Exacerbation_P2 = df["Exacerbation_Frequency_P2"]
    Exacerbation_P2[Exacerbation_P2 == ""] = "nan"
    Exacerbation_P2 = Exacerbation_P2.astype(float)
    df["Frequent_Exacerbator_P2"] = Exacerbation_P2 >= 2    

    df["Frequent_Exacerbator_P2"] = df["Frequent_Exacerbator_P2"].astype(float).astype(str) 
    df.loc[np.isnan(Exacerbation_P2), "Frequent_Exacerbator_P2"] = "nan"

    Exacerbation_P3 = df["Exacerbation_Frequency_P3"]
    Exacerbation_P3[Exacerbation_P3 == ""] = "nan"
    Exacerbation_P3 = Exacerbation_P3.astype(float)
    df["Frequent_Exacerbator_P3"] = Exacerbation_P3 >= 2
    df["Frequent_Exacerbator_P3"] = df["Frequent_Exacerbator_P3"].astype(float).astype(str) 
    df.loc[np.isnan(Exacerbation_P3), "Frequent_Exacerbator_P3"] = "nan"
    
    

    df_cov = df[covariances]
    df_cov[df_cov == ""] = "nan"
    df_cov = df_cov.astype(float)

    
    #Loading survival dataset
    df_survival = pd.read_csv("/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/MortalitySurvivalAnalysis/COPDGene_VitalStatus_SM_NS_Aug20_update11_24_2020.csv")\
        .set_index("sid")[['vital_status', 'days_followed']]
    
    df_merged = pd.merge(df_cov, df_survival,left_index=True,right_index=True, how = "outer")

    #Computing 5-year mortality
    df_merged["5-year Mortality_P1"] = 1 - np.array(  df_merged["days_followed"] > 365 * 5, float )
    df_merged.loc[ 
        np.bitwise_and( df_merged["days_followed"] <= 365 * 5, df_merged["vital_status"] == 0 ), "5-year Mortality_P1"] = np.nan

    df_merged.loc[ 
        np.bitwise_and( df_merged["days_followed"] <= 365 * 5, df_merged["vital_status"] == 1 ), "5-year Mortality_P1"] = 1    
    df_merged.loc[np.isnan(df_merged["days_followed"]), "5-year Mortality_P1" ] = np.nan
    
    N_days_P1P2 = df["days_CT1_CT2"]  
    N_days_P1P2[N_days_P1P2 == ""] = "nan"
    N_days_P1P2 = N_days_P1P2.astype(float)
   

    df_merged["days_P2"] = df_merged["days_followed"] - N_days_P1P2



    
    df_merged["5-year Mortality_P2"] = 1 - np.array(  df_merged["days_P2"] > 365 * 5, float )

    df_merged.loc[ 
        np.bitwise_and( df_merged["days_P2"] <= 365 * 5, df_merged["vital_status"] == 0 ), "5-year Mortality_P2"] = np.nan

    df_merged.loc[ 
        np.bitwise_and( df_merged["days_P2"] <= 365 * 5, df_merged["vital_status"] == 1 ), "5-year Mortality_P2"] = 1    

    df_merged.loc[np.isnan(df_merged["days_followed"]), "5-year Mortality_P2" ] = np.nan
    df_merged.loc[np.isnan(N_days_P1P2), "5-year Mortality_P2"] = np.nan
    
    # Loading the subtype dataset.
    df_subtype = pd.read_csv("/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/Subtyping/COPDGene_Subtype_Smoker.csv").set_index("sid")
    
    df_subtype = df_subtype[subtypes_used]
    
    df_merged = pd.merge(df_merged, df_subtype,left_index=True,right_index=True, how = "outer")
    
    # Computing the peel/core ratio of perc15.
    Z_C = pd.read_csv("/ocean/projects/asc170022p/shared/IEA_data/perc15_by_bands/r20+.csv")
    Z_P = pd.read_csv("/ocean/projects/asc170022p/shared/IEA_data/perc15_by_bands/r0_5.csv")

    Z_C = Z_C.set_index("Unnamed: 0").rename(columns = {"perc15":"perc15 core"})
    Z_P = Z_P.set_index("Unnamed: 0").rename(columns = {"perc15":"perc15 peel"})



    df_prec15_ratio = pd.merge(Z_C, Z_P, left_index = True, right_index = True,)
    df_prec15_ratio["perc15_ratio"] = 100 * np.emath.log(df_prec15_ratio["perc15 peel"] / df_prec15_ratio["perc15 core"] )



    df_merged = pd.merge(df_merged, df_prec15_ratio[["perc15_ratio"]], left_index = True, right_index = True, 
              how = "outer")
        
    
    return df_merged
    
def load_perc15_bands():   
    """
    Load the data of perc15 by different bands.
    """
    data_dir = "/ocean/projects/asc170022p/shared/IEA_data/perc15_by_bands/"
    file_list = os.listdir(data_dir)
    
    df = None

    for fff in file_list:

        feature_name = fff.replace(".csv", "")

        data = pd.read_csv(data_dir + fff).set_index("Unnamed: 0")
        df_fff = data.rename(columns = {"perc15": feature_name + "_perc15", "pctemph": feature_name + "_pctemph"})
        
        if df is None:
            df = df_fff
        else:
            df = pd.merge(df, df_fff, left_index = True, right_index = True)
            
    return df


def load_gene_data():
    
    """
    Load the gene expression dataset.
    """
    
    raw_data = np.loadtxt("/ocean/projects/asc170022p/shared/Data/COPDGene/GeneticGenomicData/RNAseq_freeze3/gene_counts_TMM_LC.tsv", 
                          dtype = str, delimiter = "\t")

    gene_sid = raw_data[0, :-1]
    gene_id = list(raw_data[1:, -1])

    rna_data = np.array( raw_data[1:, :-1], float ).T


    df_rna = pd.DataFrame(data = rna_data, columns = gene_id, index = gene_sid)
    df_rna["sid"] = gene_sid    



    return df_rna
    

def load_SSL_features_P1():
    
    """
    Load the self-supervised features for phase 1.
    """
    
    
    patch_rep_dir = "/ocean/projects/asc170022p/shared/IEA_data/SSL_features_P1.pickle"
    with open(patch_rep_dir, 'rb') as fp:
        patch_rep_dict = pickle.load(fp)
        
    file_list = patch_rep_dict.keys() 
    sid_used = [ iii[:6] for iii in file_list ]
    patch_rep = [ 
    np.array( patch_rep_dict[iii]['patch_rep']) 
    for iii in file_list]
    
    patch_rep = np.array(patch_rep)  
    
    return sid_used, patch_rep
    
def load_SSL_features_P2():
    """
    Load the self-supervised features for phase 2.
    """
    
    
    patch_rep_dir = "/ocean/projects/asc170022p/shared/IEA_data/SSL_features_P2.pickle"
    with open(patch_rep_dir, 'rb') as fp:
        patch_rep_dict = pickle.load(fp)
        
    file_list = patch_rep_dict.keys() 
    sid_used = [ iii[:6] for iii in file_list ]
    patch_rep = [ 
    np.array( patch_rep_dict[iii]['patch_rep']) 
    for iii in file_list]
    
    patch_rep = np.array(patch_rep)  
    
    return sid_used, patch_rep    

def load_SSL_PCs( n_components = 10 ):
    """
    Return the PCs for the self-supervised features.
    """
    
    
    M = PCA(n_components = n_components)
    sid_P2, SSL_P2 = load_SSL_features_P2()
    sid_P1, SSL_P1 = load_SSL_features_P1()
    
    SSL_P1 = SSL_P1.reshape(-1, 581 * 128)
    SSL_P2 = SSL_P2.reshape(-1, 581 * 128)
    
    M.fit(SSL_P2)
    PCs_P1 = M.transform(SSL_P1)
    PCs_P2 = M.transform(SSL_P2)
    
    df_P1 = pd.DataFrame( PCs_P1, 
                 columns = ["PC{}_P1".format(iii) for iii in range(n_components)], 
                index = sid_P1)

    df_P2 = pd.DataFrame( PCs_P2, 
                 columns = ["PC{}_P2".format(iii) for iii in range(n_components)], 
                index = sid_P2)

    df = pd.merge(df_P1, df_P2, left_index = True, right_index = True, how = "outer")
    
    return df
    
    
class image_gene_dataset(Dataset):
    """
    The data loader for image-gene data.
    """
    def __init__(self, train = True, cv = None, n_cv = 5, seed = 0, N_test = 300):
    
        np.random.seed(seed)
        
        gene_df = load_gene_data()
        self.genelist = [iii for iii in gene_df if "EN" in iii]
        image_sid, self.image_features = load_SSL_features_P2()
        
        gene_df["image_index"] = -1
        
        # match the gene data and image features according to sid by finding the index
        for iii in gene_df.index:
            if iii in image_sid:
                gene_df.loc[iii, "image_index"] = image_sid.index(iii)

        self.df = gene_df[ gene_df["image_index"] != -1 ]        
        
        
        #randomly shuffle the data and split into training and test sets
        self.rand_idx = list(self.df.index)
        np.random.shuffle(self.rand_idx)
        N_train = len(self.rand_idx) - N_test

        train_idx = self.rand_idx[:N_train]
        test_idx = self.rand_idx[N_train:]
        
        if cv is None: # Not doing cross-validation
            
            if train: # The training set is returned if train 
                self.idx_used = train_idx
            else: # the test set is returned, otherwise
                self.idx_used = test_idx
                
        else: # cross-validation
            validation_idx = train_idx[ N_train // 5 * cv : N_train // 5 * (cv + 1) ]
            if train: # The training set is returned
                self.idx_used = [iii for iii in train_idx if not iii in validation_idx]
            else: # The validation set is returned
                self.idx_used = validation_idx

        # standardize gene data according to the training set

        mean = self.df.loc[train_idx, self.genelist ].mean(0)
        std = self.df.loc[train_idx, self.genelist ].std(0)
        self.df[self.genelist] =  ( self.df[self.genelist] - mean )/std
                
        self.N = len(self.idx_used)        
        
        
    def __len__(self):
        return self.N
    
    
    def __getitem__(self, idx):
        # returns the sid, the image features and the gene data
        sid = self.idx_used[idx]
        
        return sid, \
               torch.tensor( self.image_features[ self.df.loc[sid, "image_index"] ] ).float(), \
               torch.tensor( np.array(self.df.loc[sid, self.genelist], float) ).float()
       
        
        

