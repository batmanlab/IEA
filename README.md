# Image-Expression Axes (IEAs) for COPDGene

The project jointly analyze the blood RNA-seq gene expression and CT scan images from 1,223 subjects in the COPDGene study, to identify the shared aspects of inflammation and lung structural changes that we refer to as Image-Expression Axes (IEAs). We extract the CT-images features using a context-aware self-supervised representation learning (CSRL). These features were then tested for association to gene expression levels in order to select genes for future analysis. For the subset of selected genes, we trained a deep learning model to identify IEAs that capture distinct patterns of association between CSRL features and blood gene expression. We then related these axes to cross-section COPD-related features and prospective health outcomes through regression and Cox proportional harzard models. 

We identified two distinct IEAs that capture most of the relationship between CT images and blood gene expression. IEAemph captures an emphysema-predominant process with a strong positive correlation to CT emphysema and a negative correlation to FEV1 and Body Mass Index (BMI). IEAairway captures an airways-predominant process with a positive correlation to BMI and airway wall thickness and a negative correlation to emphysema. The IEAairway axes was significantly associated with risk of mortality (HR 1.60, CI 1.13-2.25). Both axes showed a positive correlation to the emphysema peel-core distribution, defined as 100 times the logarithm of the ratio of perc15 in the lung periphery to the lung core.

# Data Analysis

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/IEA/blob/main/Figures_Tables/summary.png">
</p>




To reproduce the results follow the following steps:
### Clone the repository
```
git clone https://github.com/batmanlab/IEA.git
cd IEA
```


### Install the required packages
```
conda env create -f environment.yml -n IEA
conda activate IEA
```


### Gene selection
```
python ./src/gene_selection.py
```

### Training the model with Cross validation 
```
python ./src/train_IEA.py
```

### Summarizing the cross-validation results
```
python ./src/summarize_cv.py
```







# Tables and Figures
The primary results of the Tables and Figures can be regenerated with the folloing python notebooks:

[Table 1](./Figures_Tables/main_text/Table1.ipynb) Subject characteristics in training and test data. 

[Table 2](./Figures_Tables/main_text/Table2.ipynb) Pearson’s correlation between image-expression axes (IEAs) and COPD-related characteristics and health outcomes. 

[Table 3](./Figures_Tables/main_text/Table3.ipynb) Multivariable associations of image-expression axes (IEAs) to continuous COPD-related characteristics and health outcomes. 

[Table 4](./Figures_Tables/main_text/Table4.ipynb) Multivariable associations of image-expression axes (IEAs) to Frequent Exacerbations and Mortality. 

[Table 5](./Figures_Tables/main_text/Table5.ipynb) Characteristics of subgroups defined by diving the Image-Expression Axes (IEAs) into quadrants.

[Table 6](./Figures_Tables/main_text/Table6.ipynb) Correlation Coefficient among Image-expression Axes (IEAs), factor analysis axes (FAs)and PCA image only axes (PCA-I).

[Figure 4](./Figures_Tables/main_text/Figure4.ipynb)

[Table E1](./Figures_Tables/Supplementary/TableE1.ipynb). Pearson’s Correlation coefficients for IEAs in 5-fold cross-validation (CV).

[Table E2](./Figures_Tables/Supplementary/TableE2.ipynb) Pearson’s correlation coefficients between IEAs identified using different adjusted p-value thresholds for inclusion of genes in the IEA model.

[Table E3](./Figures_Tables/Supplementary/TableE3.ipynb) The correlation between perc15 ratio and IEAs.

[Table E4](./Figures_Tables/Supplementary/TableE4.ipynb) Multivariate analysis with image-expression axes (IEAs) and COPD measurements with COPDGene visit 2 data.

[Table E5](./Figures_Tables/Supplementary/TableE5.ipynb) Logistic regression and Cox model with image-expression axes (IEAs) and COPD measurements with COPDGene visit 2 data.

[Table E6](./Figures_Tables/Supplementary/TableE6.ipynb) Univariate analysis with image-expression axes (IEAs), phenotype disease axes and image-based features with COPDGene visit 1 data.

[Table E7](./Figures_Tables/Supplementary/TableE7.ipynb) Linear regression analysis with image-expression axes (IEAs) and factor analysis axes (FAs) on COPDGene visit 1 data.

[Table E8](./Figures_Tables/Supplementary/TableE8.ipynb) Multivariate analysis with image-expression axes (IEAs) and factor analysis axes (FAs) on COPDGene visit 1 data.

[Table E9](./Figures_Tables/Supplementary/TableE9.ipynb)  Linear regression analysis with image-expression axes (IEAs) and PCA Image Only Axes (PCA-I) on COPDGene visit 1 data.

[Table E10](./Figures_Tables/Supplementary/TableE10.ipynb) Logistic regression and Cox model with image-expression axes (IEAs) and PCA Image Only Axes (PCA-I) on COPDGene visit 1 data.

[Figure E2](./Figures_Tables/Supplementary/FigureE2.ipynb) Mean R2 vs. number of IEAs. 

[Figure E4](./Figures_Tables/Supplementary/FigureE4.ipynb) Histograms of the variance explained by the IEAs and PCA-I. 




