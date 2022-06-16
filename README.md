# SPH029 - Machine Learning
MSc Health Data Analytics w/ Machine Learning

Data source: [A merged lung cancer transcriptome dataset for clinical predictive modelling - Su Bin Lim et al. (2018)](https://www.nature.com/articles/sdata2018136)


## Overview
The aim of this project is to explore the use of machine learning for the prediction of lung cancer. In particular, I will apply different machine learning techniques on microarray expression levels from a lung cancer transcriptome dataset[^1] in order to classify Non-small cell lung cancer (NSCLC) cases. The project broadly consists of two main tasks - an unsupervised task and a supervised task - which are described, respectively, as follows:
1. Identify the NSCLC disease subtypes using the expression levels
2. Compare and contrast the performance of classic machine learning algorithms for the classification of NSCLC cases

Within this repository is two separate attempts at completing the above objectives - namely '_Attempt 1_' and '_Attempt 2_':

#### Attempt 1
My first Python machine learning project during the Machine Learning module of the MSc Health Data Analytics course (taught by Dr Seth Flaxman).

**Unsupervised Learning**
- Principal Component Analysis (PCA)
- t-distributed Stochastic Neighbor Embedding (t-SNE)

**Supervised Learning**
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbor (kNN)

#### Attempt 2
A personal project to showcase a more complete body of work and further develop skills that I have gained since the first attempt. This aims to investigate a different set of machine learning models, albeit more rigorously. 


- **Unsupervised Learning**
  - Principal Component Analysis (PCA)
  - K-Means clustering
  - **D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise (DBSCAN)
  - **O**rdering **P**oints **T**o **I**dentify the **C**lustering **S**tructure (OPTICS)

- **Supervised Learning**
  - Logistic Regression
  - Partial Least Squares Discriminant Analysis (PLS-DA)
  - Multilayer Perceptron (MLP)???
  - Convolutional Neural Network (CNN)???




[^1]: https://www.nature.com/articles/sdata2018136
