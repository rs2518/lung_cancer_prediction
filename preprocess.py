import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, SparsePCA

## ONLY USE AFTER THOROUGHLY CHECKING WARNINGS
# import warnings

# warnings.filterwarnings("ignore")

## TODO: 'Prettify' seaborn plots


# =============================================================================
# PREPROCESS DATA
# =============================================================================

# Load data
# ---------
combat_filter_expr_url = "https://figshare.com/ndownloader/files/9194527"
combat_nofilter_expr_url = "https://figshare.com/ndownloader/files/9194533"

# # Uncorrected, unfiltered expression levels
# nocombat_nofilter_expr_url = "https://figshare.com/ndownloader/files/9194530"

# # Microarray differential expressed (DE) gene analysis
# microarray_logfc_url = "https://figshare.com/ndownloader/files/9194536"

# # TCGA differential expressed (DE) gene analysis
# tcga_logfc_url = "https://figshare.com/ndownloader/files/9194539"

# # TCGA database expression levels
# tcga_expr_url = "https://figshare.com/ndownloader/files/9194545"

# # DE genes by rank (Microarray vs. TCGA)
# combined_rank_url = "https://figshare.com/ndownloader/files/9194542"

clinical_info_url = "https://figshare.com/ndownloader/files/10449075"


combat_filter_expr = pd.read_csv(combat_filter_expr_url, sep="\t").T
combat_nofilter_expr = pd.read_csv(combat_nofilter_expr_url, sep="\t").T
# nocombat_nofilter_expr = pd.read_csv(nocombat_nofilter_expr_url, sep="\t").T
# microarray_logfc = pd.read_csv(microarray_logfc_url, sep="\t")
# tcga_logfc = pd.read_csv(tcga_logfc_url, sep="\t")
# tcga_expr = pd.read_csv(tcga_expr_url, sep="\t", index_col=0).T
# combined_rank = pd.read_csv(combined_rank_url, sep="\t")
clinical_info = pd.read_csv(clinical_info_url, sep="\t", index_col=0)

cols = {"Histology (ADC: adenocarcinoma; "
        "LCC: large cell carcinoma; "
        "SCC: squamous cell carcinoma)" : "Histology",
        
        "Disease Status (NSCLC: primary tumors; "
        "Normal: non-tumor lung tissues)" : "Disease Status",
        
        "Overall survival (0: alive 1:deceased)" : "Survival",
        
        "Overall survival (month)" : "Survival Month"}
clinical_info.rename(columns=cols, inplace=True)


# Merge data
# ----------
df = clinical_info.merge(combat_filter_expr, how="inner",
                         left_index=True, right_index=True)

threshold = math.floor(0.7*df.shape[0])
data = df.dropna(axis=1, how="any", thresh=threshold)
data_cc = data.dropna(axis=0, how="any")



# =============================================================================
# VALIDATE DATA
# =============================================================================

# PCA plots
# ---------
seed = 9610

# Check batch-corrected (PCA score plot). hue = dataset
pca_corrected = PCA(random_state=seed)
pca_corrected.fit(combat_filter_expr)

cols = ["Dataset", "Disease Status", "Gender", "Histology"]
pc = pca_corrected.transform(combat_filter_expr)
pc = pd.DataFrame(pc, index=df.index,
                  columns=["PC"+str(i+1) for i in range(pc.shape[0])])
test = pd.concat([df[cols], pc], axis=1)

## TODO: Recode missing gender as "unknown"
## TODO: Recode "stage" features which have additional spaces
## TODO: Recode "Histology" features

## TODO: Set markers and colors to be consistent with literature


# By dataset
fig, ax = plt.subplots()
sns.scatterplot(x="PC1", y="PC2", data=test, hue="Dataset", ax=ax)
fig.show()

# By disease status
fig, ax = plt.subplots()
sns.scatterplot(x="PC1", y="PC2", data=test, hue="Disease Status", ax=ax)
fig.show()

# By subtype
fig, ax = plt.subplots()
sns.scatterplot(x="PC1", y="PC2", data=test, hue="Disease Status", ax=ax)
fig.show()

# By gender
fig, ax = plt.subplots()
sns.scatterplot(x="PC1", y="PC2", data=test, hue="Gender", ax=ax)
fig.show()



# Differences in signs for components when comparing R to Python (PC2 to be
# flipped in this case)
# Need to check documentation or see PLS code for help


# Check batch-corrected (PCA score plot). hue = gender (impute missing gender)




# Check uncorrected (PCA score plot)
pca_uncorrected = PCA(random_state=seed)
pca_uncorrected.fit(combat_filter_expr)


# Sparse PCA plots
# ----------------

# Check batch-corrected (sPCA score plot). hue = dataset
spca_corrected = PCA(random_state=seed)
spca_corrected.fit(combat_filter_expr)