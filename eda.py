import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from lcml import ROOT
from lcml import (CLINICAL_INFO_COLS,
                  STATUS_MAP,
                  HISTOLOGY_MAP,
                  STAGE_MAP,
                  SMOKING_MAP)
from lcml import create_directory


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def missing_data_prop(data, return_counts=False, sort=None):
    """Proportion of missing values by column
    """
    d = {"ascending":True, "descending":False}
    
    if return_counts:
        n = 1
    else:
        n = data.shape[0]
    
    if sort is None:
        return np.sum(data.isna(), axis=0)/n
    else:
        tab = np.sum(data.isna(), axis=0)/n
        return tab.sort_values(ascending=d[sort])
    
def check_filtered_records(df1, df2, col, criteria):
    """Check if df1 and df2 have the same number of records after applying
    filter
    """
    a = len(df1[df1[col]==criteria])
    b = len(df2[df2[col]==criteria])
    return a==b

def group_counts(data, col, return_counts=False):
    """Get counts across grouping factor
    """
    columns = data.columns.to_list()
    columns.remove(col)
    desc = data.groupby(col).describe(include="all")

    tab = \
        pd.DataFrame(np.array([desc[(c, "count")].values for c in columns]).T,
                     index=desc.index, columns=columns)
        
    if return_counts:
        return tab
    else:
        return tab/tab["Disease Status"].values.reshape(-1,1)



# =============================================================================
# LOAD DATASETS
# =============================================================================

# Patient status by study
patient_status_url = "https://figshare.com/ndownloader/files/9194524"

# Batch-corrected, filtered expression levels
combat_filter_expr_url = "https://figshare.com/ndownloader/files/9194527"

# Batch-corrected, unfiltered expression levels
combat_nofilter_expr_url = "https://figshare.com/ndownloader/files/9194533"

# Uncorrected, unfiltered expression levels
nocombat_nofilter_expr_url = "https://figshare.com/ndownloader/files/9194530"

# Microarray differential expressed (DE) gene analysis
microarray_logfc_url = "https://figshare.com/ndownloader/files/9194536"

# TCGA differential expressed (DE) gene analysis
tcga_logfc_url = "https://figshare.com/ndownloader/files/9194539"

# TCGA database expression levels
tcga_expr_url = "https://figshare.com/ndownloader/files/9194545"

# DE genes by rank (Microarray vs. TCGA)
combined_rank_url = "https://figshare.com/ndownloader/files/9194542"

# Clinical data
clinical_info_url = "https://figshare.com/ndownloader/files/10449075"



# # Load data
# # ---------
# patient_status = pd.read_csv(patient_status_url, index_col=0)
combat_filter_expr = pd.read_csv(combat_filter_expr_url, sep="\t").T
combat_nofilter_expr = pd.read_csv(combat_nofilter_expr_url, sep="\t").T
nocombat_nofilter_expr = pd.read_csv(nocombat_nofilter_expr_url, sep="\t").T
microarray_logfc = pd.read_csv(microarray_logfc_url, sep="\t")
# tcga_logfc = pd.read_csv(tcga_logfc_url, sep="\t")
# tcga_expr = pd.read_csv(tcga_expr_url, sep="\t", index_col=0).T
combined_rank = pd.read_csv(combined_rank_url, sep="\t")
clinical_info = pd.read_csv(clinical_info_url, sep="\t", index_col=0,
                            header=0, names=CLINICAL_INFO_COLS)



# =============================================================================
# SANITY CHECKS
# =============================================================================

# # Check clinical_info and patient_status records match
# merge = pd.merge(patient_status, clinical_info,
#                   how="inner", left_index=True, right_index=True)    
# if clinical_info.shape[0] != merge.shape[0]:
#     raise ValueError("Clinical information and patient status "
#                       "records do not match!")

# # Check "Gene" and "Gene.1" columns match in rank table
# if all(combined_rank["Gene"]!=combined_rank["Gene.1"]):
#     raise ValueError("'Gene' does not match 'Gene.1' in rank table!")
    


# =============================================================================
# EXAMINE DATA
# =============================================================================

# # Missing data
# # ------------
# sort="descending"
# na_patient_status = missing_data_prop(patient_status, sort=sort)
# na_combat_filter = missing_data_prop(combat_filter_expr, sort=sort)
# na_combat_nofilter = missing_data_prop(combat_nofilter_expr, sort=sort)
# na_nocombat_nofilter = missing_data_prop(nocombat_nofilter_expr, sort=sort)
# na_microarray_logfc = missing_data_prop(microarray_logfc, sort=sort)
# na_tcga_logfc = missing_data_prop(tcga_logfc, sort=sort)
# na_tcga_expr = missing_data_prop(tcga_expr, sort=sort)
# na_combined_rank = missing_data_prop(combined_rank, sort=sort)
# na_clinical_info = missing_data_prop(clinical_info)
# # NOTE: Only clinical information contains missing values

# # Counts by dataset
# dataset_counts = group_counts(clinical_info, col="Dataset",
#                               return_counts=True)
# status_counts = group_counts(clinical_info, col="Disease Status",
#                              return_counts=True)
# histology_counts = group_counts(clinical_info, col="Histology",
#                                 return_counts=True)
# gender_counts = group_counts(clinical_info, col="Gender",
#                              return_counts=True)
# stage_counts = group_counts(clinical_info, col="Stage",
#                             return_counts=True)
# surv_counts = group_counts(clinical_info, col="Survival",
#                            return_counts=True)
# smoking_count = group_counts(clinical_info, col="Smoking",
#                              return_counts=True)
# # NOTE: Most missing clinical data is based on the study they were taken from


# # Checking feature values
# # -----------------------
# # Scan through non-numeric feature classes
# stage = clinical_info[~clinical_info["Stage"].isnull()]
# histology = clinical_info[~clinical_info["Histology"].isnull()]
# surv_month = clinical_info[~clinical_info["Survival Month"].isnull()]
# smoking = clinical_info[~clinical_info["Smoking"].isnull()]
# # NOTE: "Stage", "Histology" and "Smoking" need to be recoded.
# # "Survival Month" should be removed


# # Inspect numeric features
# desc = clinical_info.describe()
# # NOTE: No peculiar values


# Preliminary cleaning of data
# ----------------------------
# Recode "Disease Status"
clinical_info["Disease Status"] = clinical_info["Disease Status"].map(
    STATUS_MAP)
# NSCLC/Normal --> Tumour/Non-tumour

# Recode "Stage"
clinical_info["Stage_opt"] = clinical_info["Stage"].apply(
    lambda x: 1 if x==" pT2N0" else STAGE_MAP[x])
# Optimistic recoding of "Stage" (pT2N0 --> 1)
clinical_info["Stage_pes"] = clinical_info["Stage"].apply(
    lambda x: 2 if x==" pT2N0" else STAGE_MAP[x])
# Pessimistic recoding of "Stage" (pT2N0 --> 2)

# # Recode "Stage"
# clinical_info["Stage_opt"] = clinical_info["Stage"].apply(
#     lambda x: "1B" if x==" pT2N0" else STAGE_MAP[x])
# # Optimistic recoding of "Stage" (pT2N0 --> 1B)
# clinical_info["Stage_pes"] = clinical_info["Stage"].apply(
#     lambda x: "2A" if x==" pT2N0" else STAGE_MAP[x])
# # Pessimistic recoding of "Stage" (pT2N0 --> 2A)

# Recode "Smoking"
clinical_info["Smoking_opt"] = clinical_info["Smoking"].apply(
    lambda x: "Never" if x=="Non-smoking" else SMOKING_MAP[x])
# Optimistic recoding of "Smoking" (non-smoking --> never)
clinical_info["Smoking_pes"] = clinical_info["Smoking"].apply(
    lambda x: "Former" if x=="Non-smoking" else SMOKING_MAP[x])
# Pessimistic recoding of "Smoking" (non-smoking --> former)

# Recode "Histology"
clinical_info.loc[(clinical_info["Disease Status"]=="Non-tumour")
    & (clinical_info["Histology"].isnull()), "Histology"] = "Healthy"            
# Disease subtype for healthy patients should be "Healthy"

clinical_info["Histology"] = clinical_info["Histology"].map(HISTOLOGY_MAP)

# Remove columns
rm = ["Race",
      "Survival Month",
      "Recurrence",
      "Others",
      "TNM stage (T)",
      "TNM stage (N)",
      "TNM stage (M)"]
clinical_info.drop(columns=rm, inplace=True)

# Replace NAs for plots
cols = ["Stage_opt", "Stage_pes", "Smoking_opt", "Smoking_pes", "Histology"]
clinical_info.loc[:, cols] = clinical_info.loc[:, cols].fillna("Missing")



# =============================================================================
# EXPLORATORY PLOTS
# =============================================================================

figpath = os.path.join(ROOT, "figures", "exploratory")
create_directory(figpath)


# Plot age distributions
# ----------------------
## TODO: CONSIDER DIFFERENT 'STYLES' ACROSS POSSIBLE CONFOUNDERS.
# ADD ANNOTATIONS (MEDIAN BY HUE)
## TODO: SET COLORMAP FOR CATEGORIES
hues = ["Dataset", "Disease Status", "Gender", "Histology"]

for x, hue in [("Age", hue) for hue in hues]:
    title = x+" by "+hue.lower()
    
    fig, ax = plt.subplots(figsize=(8,8))
    sns.kdeplot(data=clinical_info, x=x, hue=hue,
                ax=ax, warn_singular=False)
    # fig.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    ax.set_title(title)
    fig.tight_layout()
    
    name = title.lower().replace(" ", "_")
    path = os.path.join(figpath, name+".png")
    fig.savefig(path, bbox_inches="tight")
    

# "Optimistic" vs. "Pessimistic" estimates
hues = ["Stage", "Smoking"]
for x, hue in [("Age", hue) for hue in hues]:
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,8))
    title = x+" by "+hue.lower()
    fig.suptitle(title)
    for i, est in enumerate(["Optimistic", "Pessimistic"]):
        title = x+" by "+hue.lower()+" ("+est+")"
        subhue = hue+"_"+est.lower()[0:3]
        
        sns.kdeplot(data=clinical_info, x=x, hue=subhue,
                    ax=ax[i], warn_singular=False)
        # fig.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
        ax[i].set_title(est)
        fig.tight_layout()
    
        name = x+" by "+subhue
        path = os.path.join(figpath, name.lower().replace(" ", "_")+".png")
        fig.savefig(path, bbox_inches="tight")

## TODO: CREATE CONTINGENCY TABLES (PD.CROSSTAB)
## TODO: CONSIDER CHI-SQUARED TO ANALYSE CATEGORICAL FEATURES
        
     
# Recreate plots from literature
# ------------------------------
# PCA plots (Batch uncorrected vs. corrected)
n_components=3

pca_nocombat = PCA(n_components=n_components)
pca_nocombat.fit(nocombat_nofilter_expr)
X_pc_nocombat = pca_nocombat.transform(nocombat_nofilter_expr)
X_pc_nocombat = pd.DataFrame(
    X_pc_nocombat, index=nocombat_nofilter_expr.index,
    columns=["PC"+str(i+1) for i in range(n_components)])
X_pc_nocombat = pd.merge(clinical_info, X_pc_nocombat, how="inner",
                         left_index=True, right_index=True)

pca_combat = PCA(n_components=n_components)
pca_combat.fit(combat_nofilter_expr)
X_pc_combat = pca_combat.transform(combat_nofilter_expr)
X_pc_combat = pd.DataFrame(
    X_pc_combat, index=combat_nofilter_expr.index,
    columns=["PC"+str(i+1) for i in range(n_components)])
X_pc_combat = pd.merge(clinical_info, X_pc_combat, how="inner",
                       left_index=True, right_index=True)

# Flip signs to be consistent with literature
X_pc_nocombat["PC1"] = -X_pc_nocombat["PC1"]
X_pc_combat["PC2"] = -X_pc_combat["PC2"]


# By dataset
hues = sorted(set(clinical_info["Dataset"]))
colors = ["cyan", "brown", "magenta", "yellow", "orange",
          "lime", "blue", "black", "red", "lightgray"]
kwargs = {"hue_order":hues,
          "palette":dict(zip(hues, colors))}

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
sns.scatterplot(x="PC1", y="PC2", data=X_pc_nocombat, hue="Dataset",
                style="Dataset", legend=False, ax=axes[0], **kwargs)
axes[0].set_title("Batch-effect uncorrected")
axes[0].set(xlabel=None, ylabel=None)

sns.scatterplot(x="PC1", y="PC2", data=X_pc_combat, hue="Dataset",
                style="Dataset", ax=axes[1], **kwargs)
axes[1].set_title("Batch-effect corrected")
axes[1].set(xlabel=None, ylabel=None)
axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left",
               borderaxespad=0., title="Dataset")

fig.savefig(os.path.join(figpath, "2d_pca_by_dataset.png"),
            bbox_inches="tight")


# By disease status
hues = sorted(set(clinical_info["Disease Status"]))
colors = ["lime", "red"]
kwargs = {"hue_order":hues,
          "palette":dict(zip(hues, colors))}

fig, ax = plt.subplots(figsize=(7, 7))
sns.scatterplot(x="PC1", y="PC2", data=X_pc_combat, hue="Disease Status",
                ax=ax, **kwargs)
ax.set_title("Batch-effect corrected")
ax.set(xlabel=None, ylabel=None)
ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left",
          borderaxespad=0., title="Status")

fig.savefig(os.path.join(figpath, "2d_pca_by_status.png"),
            bbox_inches="tight")


# Additional exploratory plots
# ----------------------------
# PCA by Histology
hues = ["ADC", "ASC", "LCC", "LCNEC", "NFA",
        "SCC", "Mixed", "Healthy", "Other", "Missing"]
colors = ["blue", "fuchsia", "yellow", "orange", "cyan",
          "red", "blueviolet", "lime", "black", "lightgray"]
kwargs = {"hue_order":hues,
          "palette":dict(zip(hues, colors))}

fig, ax = plt.subplots(figsize=(7, 7))
sns.scatterplot(x="PC1", y="PC2", data=X_pc_combat, hue="Histology",
                ax=ax, **kwargs)
ax.set_title("Batch-effect corrected")
ax.set(xlabel=None, ylabel=None)
ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left",
          borderaxespad=0., title="Histology")

fig.savefig(os.path.join(figpath, "2d_pca_by_histology.png"),
            bbox_inches="tight")


# Plot expression levels
# ----------------------
rs = check_random_state(1010)
s = sorted(rs.randint(0, combat_filter_expr.shape[1]-1, 9))
combat_filter_expr.iloc[:, s].hist(figsize=(10,10), density=True)
plt.suptitle("Batch corrected expression levels", fontsize=16)

np.log(combat_filter_expr.iloc[:, s]).hist(figsize=(10,10), density=True)
plt.suptitle("Batch corrected log-transformed expression levels", fontsize=16)
# NOTE: Expression levels appear more normally distributed after
# log transformation


# Plot correlation
# ----------------
# Plot clustermaps for differentially expressed genes
genes = sorted(microarray_logfc[microarray_logfc["threshold"]!=False].index)
if any([g not in combat_filter_expr.columns.to_list() for g in genes]):
    raise ValueError("Not all differentially expressed genes appear "
                     "in filtered gene list!")

df_expr = combat_filter_expr.loc[:, genes]
corr = df_expr.corr()
# WARNING: Correlation matrix for all genes takes around 20 mins

# Gene-gene correlation
fig = sns.clustermap(corr, cmap="vlag", vmin=-1, vmax=1, figsize=(14,14))
fig.savefig(os.path.join(figpath, "gene_correlation.png"), bbox_inches="tight")
# NOTE: Strong correlation and block structure among most significant genes
# Irrepresentable condition isn't valid (penalisation should work)


# Clustered observations by "Disease Status"
tumour = clinical_info.loc[df_expr.index, "Disease Status"]
cmap = dict(zip(sorted(set(tumour)), ["lime", "red"]))
fig = sns.clustermap(df_expr, figsize=(14,14), cmap="Spectral",
                     row_colors=tumour.map(cmap))
fig.savefig(os.path.join(figpath, "status_clustermap.png"),
            bbox_inches="tight")


# Clustered observations by "Histology"
histology = clinical_info.loc[df_expr.index, "Histology"]
colors = ["blue", "fuchsia", "lime", "yellow", "orange",
          "lightgray", "blueviolet", "cyan", "black", "red"]
cmap = dict(zip(sorted(set(histology)), colors))

fig = sns.clustermap(df_expr, figsize=(14,14), cmap="Spectral",
                     row_colors=histology.map(cmap))
fig.savefig(os.path.join(figpath, "histology__clustermap.png"),
            bbox_inches="tight")

## TODO: REPEAT CORRELATION PLOTS ON LOG TRANSFORMED VALUES AND COMPARE
# NOTE THAT CLUSTERMAPS CANNOT BE VIEWED SIDE-BY-SIDE IN SUBPLOTS
## TODO: CONSIDER ADDING COLUMN COLOUR (UPREGULATED VS DOWNREGULATED GENES)