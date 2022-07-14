# Image-Expression Axes (IEAs) for COPDGene

The project jointly analyze the blood RNA-seq gene expression and CT scan images from 1,223 subjects in the COPDGene study, to identify the shared aspects of inflammation and lung structural changes that we refer to as Image-Expression Axes (IEAs). We extract the CT-images features using a context-aware self-supervised representation learning (CSRL). These features were then tested for association to gene expression levels in order to select genes for future analysis. For the subset of selected genes, we trained a deep learning model to identify IEAs that capture distinct patterns of association between CSRL features and blood gene expression. We then related these axes to cross-section COPD-related features and prospective health outcomes through regression and Cox proportional harzard models. 
 
We identified two distinct IEAs that capture most of the relationship between CT images and blood gene expression. IEA<sub>emph</sub> captures an emphysema-predominant process with a strong positive correlation to CT emphysema and a negative correlation to FEV1 and Body Mass Index (BMI). IEA<sub>airway</sub> captures an airways-predominant process with a positive correlation to BMI and airway wall thickness and a negative correlation to emphysema. The IEA<sub>airway</sub> was significantly associated with risk of mortality (HR 2.23, CI 1.49-3.35). Both axes showed a positive correlation to the emphysema peel-core distribution, defined as the logarithm of the ratio of perc15 in the lung periphery to the lung core.

# Data Analysis

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/IEA/blob/main/Figures_Tables/summary.png">
</p>

The data analysis can be separated into the following 4 parts: 1) Register and patchify the CT scans; 2) Extact the features from the processed CT scans; 3) Select genes based on the association between gene expression and extracted image features; 4) Train the deep learning model that identify IEA using the image features and the expression levels of the selected genes. We provide the code for step 3 and 4 in this repository.

# Generate the IEAs
The IEAs can be generated with the following steps:
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
### Model training 
```
python ./src/train_IEA.py
```

The user can skip model training and download the pre-train models, by running the following code:
```
curl -L "https://docs.google.com/uc?export=download&id=1qeMC8y2jRU7iI0raWoT1YJZktNqT0S-y" --output primary_models.zip
unzip -o primary_models.zip
```

### Summarizing the cross-validation results
```
python ./src/summarize_cv.py
```

# Additional model training 
To generate the supplemental Tables E1 and E2, it is required to variate the thresholds of the adjusted p-values for gene selection and train the IEA models with different sets of selected genes. To train these models, run the following script:
```
chmod +x ./src/gene_thresholds.sh
./src/gene_thresholds.sh 
```
To generate the supplemental Figure E2, it is required to variate the number of IEAs when training the IEA models. To train these models, run the following script:
```
chmod +x ./src/num_IEAs.sh
./src/num_IEAs.sh 
```
These two scripts might take long to run, and it is recommended to parallelize these scripts.

The user can skip model training and download the pre-trained models with the following code:
```
curl -L "https://docs.google.com/uc?export=download&id=10-JQ3R4hJmC1nXhzucedr2hMAFkOHoHn" --output models.zip
unzip -o models.zip
```



# Tables and Figures
The primary results of the Tables and Figures can be regenerated with the folloing python notebooks:

[Table 1](./Figures_Tables/main_text/Table1.ipynb) Subject characteristics in training and test data. 

[Table 2](./Figures_Tables/main_text/Table2.ipynb) Pearson correlation coefficients between image-expression axes (IEAs) and COPD-related characteristics and health outcomes. 

[Table 3](./Figures_Tables/main_text/Table3.ipynb) Multivariable associations of image-expression axes (IEAs) to continuous COPD-related characteristics and health outcomes. 

[Table 4](./Figures_Tables/main_text/Table4.ipynb) Multivariable associations of image-expression axes (IEAs) to Frequent Exacerbations and Mortality. 

[Table 5](./Figures_Tables/main_text/Table5.ipynb) Characteristics of subgroups defined by diving the Image-Expression Axes (IEAs) into quadrants.

[Table 6](./Figures_Tables/main_text/Table6.ipynb) Correlation Coefficient among Image-expression Axes (IEAs), factor analysis axes (FAs)and PCA image only axes (PCA-I).

[Figure 4](./Figures_Tables/main_text/Figure4.ipynb) Distribution of IEA<sub>emph</sub> and IEA<sub>airway</sub> values grouped by previously published COPD K-means clustering subtypes.

[Table E1](./Figures_Tables/Supplementary/TableE1.ipynb) Pearson correlation coefficients for IEAs in 5-fold cross-validation (CV).

[Table E2](./Figures_Tables/Supplementary/TableE2.ipynb) Pearson correlation coefficients between IEAs identified using different adjusted p-value thresholds for inclusion of genes in the IEA model.

[Table E3](./Figures_Tables/Supplementary/TableE3.ipynb) The correlation between perc15 ratio and IEAs.

[Table E4](./Figures_Tables/Supplementary/TableE4.ipynb) Multivariable associations of image-expression axes (IEAs) to continuous COPD-related characteristics and health outcomes. 

[Table E5](./Figures_Tables/Supplementary/TableE5.ipynb) Multivariable associations of image-expression axes (IEAs) to Frequent Exacerbators and Mortality. 

[Table E6](./Figures_Tables/Supplementary/TableE6.ipynb) Pearson correlation coefficients among image-expression axes (IEAs), factor analysis axes (FAs) and PCA image only axes (PCA-I), COPD-related characteristics and health outcomes. 

[Table E7](./Figures_Tables/Supplementary/TableE7.ipynb) Multivariable associations of image-expression axes (IEAs) and factor analysis axes (FAs) to continuous COPD-related characteristics and health outcomes. 

[Table E8](./Figures_Tables/Supplementary/TableE8.ipynb) Multivariable associations of image-expression axes (IEAs) and factor analysis axes (FAs) to Frequent Exacerbators and Mortality. 


[Table E9](./Figures_Tables/Supplementary/TableE9.ipynb)  Multivariable associations of image-expression axes (IEAs) and PCA Image Only Axes (PCA-I) to continuous COPD-related characteristics and health outcomes. 

[Table E10](./Figures_Tables/Supplementary/TableE10.ipynb) Multivariable associations of image-expression axes (IEAs) and PCA Image Only Axes (PCA-I) to Frequent Exacerbators and Mortality. 

[Figure E2](./Figures_Tables/Supplementary/FigureE2.ipynb) Variance of gene expression explained by IEAs as we variate the number of IEAs. 

[Figure E4](./Figures_Tables/Supplementary/FigureE4.ipynb) Histograms for the variance of gene expression explained by the IEAs and PCA-I. 




