# Image-Expression Axes (IEAs) for COPDGene

The project jointly analyzes the blood RNA-seq gene expression and CT scan images from 1,223 subjects in the COPDGene study to identify the shared aspects of inflammation and lung structural changes that we refer to as Image-Expression Axes (IEAs). We extract the CT-images features using context-aware self-supervised representation learning (CSRL). These features were then tested for association with gene expression levels to select genes for future analysis. For the subset of selected genes, we trained a deep-learning model to identify IEAs that capture distinct patterns of association between CSRL features and blood gene expression. We then related these axes to cross-section COPD-related features and prospective health outcomes through regression and Cox proportional hazard models.

We identified two distinct IEAs that capture most of the relationship between CT images and blood gene expression: IEA<sub>emph</sub> captures an emphysema-predominant process with a strong positive correlation to CT emphysema and a negative correlation to FEV<sub>1</sub> and Body Mass Index (BMI); IEA<sub>airway</sub> captures an airway-predominant process with a positive correlation to BMI and airway wall thickness and a negative correlation to emphysema. Pathway enrichment analysis identified 29 and 13 pathways significantly associated with IEAemph and IEAairway, respectively (adjusted p<0.001). 

# Data Analysis

<p align="center">
  <img width="75%" height="%75" src="https://github.com/batmanlab/IEA/blob/main/Figures_Tables/summary.png">
</p>

The data analysis can be separated into the following 4 parts: 1) Register and patchify the CT scans; 2) Extract the features from the processed CT scans; 3) Select genes based on the association between gene expression and extracted image features; 4) Train the deep learning model that identifies IEA using the image features and the expression levels of the selected genes. We provide the code for steps 3 and 4 in this repository. The code for steps 1 and 2 is given in [https://github.com/batmanlab/Context_Aware_SSL](https://github.com/batmanlab/Context_Aware_SSL). 

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
To generate the [Supplemental Table E1](./Figures_Tables/Supplementary/TableE1.ipynb), it is required to variate the thresholds of the adjusted p-values for gene selection and train the IEA models with different sets of selected genes. To train these models, run the following script:
```
chmod +x ./src/gene_thresholds.sh
./src/gene_thresholds.sh 
```
To generate [Figure 2](./Figures_Tables/main_text/Figure2.ipynb), it is required to variate the number of IEAs when training the IEA models. To train these models, run the following script:
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

[Figure 2](./Figures_Tables/main_text/Figure2.ipynb) Variance of gene expression explained by IEAs as we variate the number of IEAs.

[Figure 4](./Figures_Tables/main_text/Figure4.ipynb) Distribution of IEA<sub>emph</sub> and IEA<sub>airway</sub> values grouped by previously published COPD K-means clustering subtypes.

[Figure 5](./Figures_Tables/main_text/Figure5.ipynb) Histograms for the variance of gene expression explained by the IEAs and PCA-Is.


[Table E1](./Figures_Tables/Supplementary/TableE1.ipynb) Cross-validation performance in IEA training. 

[Table E2](./Figures_Tables/Supplementary/TableE2.ipynb) Linear Regression with image-expression axes (IEAs) and COPD measurements with COPDGene visit 2 data. 

[Table E3](./Figures_Tables/Supplementary/TableE3.ipynb) Logistic regression and Cox proportional harzard models with image-expression axes (IEAs) and COPD measurements with COPDGene visit 2 data. 

[Table E4](./Figures_Tables/Supplementary/TableE4.ipynb) Pearson’s correlation between image-expression axes (IEAs) and COPD-related characteristics and health outcomes, measured on 1,527 subjects from another subset of the COPDGene dataset that had not been used for model training.  

[Table E5](./Figures_Tables/Supplementary/TableE5.ipynb) Linear Regression with image-expression axes (IEAs) and COPD measurements with 1,527 subjects from another subset of the COPDGene dataset that had not been used for model training.   

[Table E6](./Figures_Tables/Supplementary/TableE6.ipynb) Logistic regression and Cox proportional harzard models with image-expression axes (IEAs) and COPD measurements with 1,527 subjects from another subset of the COPDGene dataset that had not been used for model training.  

[Table E7](./Figures_Tables/Supplementary/TableE7.ipynb) Characteristics of subgroups defined by diving the Image-Expression Axes (IEAs) into quadrants. 

[Table E8](./Figures_Tables/Supplementary/TableE8.ipynb) Covariances between Image-expression Axes (IEAs), factor analysis axes (FAs), and PCA image-only axes (PCA-Is) on COPDGene visit 1 data. 

[Table E9](./Figures_Tables/Supplementary/TableE9.ipynb) Pearson correlation coefficients among image-expression axes (IEAs), factor analysis axes (FAs) and PCA image only axes (PCA-Is), COPD-related characteristics and health outcomes. 

[Table E10](./Figures_Tables/Supplementary/TableE10.ipynb) Linear regression analysis with image-expression axes (IEAs) and factor analysis axes (FAs) on COPDGene visit 1 data.

[Table E11](./Figures_Tables/Supplementary/TableE11.ipynb) Logistic regression and Cox model with image-expression axes (IEAs) and factor analysis axes (FAs) on COPDGene visit 1 data. 

[Table E12](./Figures_Tables/Supplementary/TableE12.ipynb) Linear regression analysis with image-expression axes (IEAs) and PCA Image Only Axes (PCA-Is) on COPDGene visit 1 data. 

[Table E13](./Figures_Tables/Supplementary/TableE13.ipynb) Logistic regression and Cox model with image-expression axes (IEAs) and PCA Image Only Axes (PCA-Is) on COPDGene visit 1 data. 

[Table E18](./Figures_Tables/Supplementary/TableE18.ipynb) The correlation between perc15 ratio and IEAs.



