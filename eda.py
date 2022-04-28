import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lcml import ROOT
from lcml import (CLINICAL_INFO_COLS,
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



# Load data
# ---------
# patient_status = pd.read_csv(patient_status_url, index_col=0)
# combat_filter_expr = pd.read_csv(combat_filter_expr_url, sep="\t").T
# combat_nofilter_expr = pd.read_csv(combat_nofilter_expr_url, sep="\t").T
# nocombat_nofilter_expr = pd.read_csv(nocombat_nofilter_expr_url, sep="\t").T
# microarray_logfc = pd.read_csv(microarray_logfc_url, sep="\t")
# tcga_logfc = pd.read_csv(tcga_logfc_url, sep="\t")
# tcga_expr = pd.read_csv(tcga_expr_url, sep="\t", index_col=0).T
# combined_rank = pd.read_csv(combined_rank_url, sep="\t")
clinical_info = pd.read_csv(clinical_info_url, sep="\t", index_col=0,
                            header=0, names=CLINICAL_INFO_COLS)



# =============================================================================
# SANITY CHECKS
# =============================================================================

# # Check clinical_info and patient_status records match
# merge = pd.merge(patient_status, clinical_info,
#                   how="inner", left_index=True, right_index=True)    
# print(clinical_info.shape[0] == merge.shape[0])

# # Check "Gene" and "Gene.1" columns match in rank table
# print(all(combined_rank["Gene"]==combined_rank["Gene.1"]))



# =============================================================================
# EXAMINE DATA
# =============================================================================

# Missing data
# ------------
# # sort="descending"
# na_patient_status = missing_data_prop(patient_status, sort=sort)
# na_combat_filter = missing_data_prop(combat_filter_expr, sort=sort)
# na_combat_nofilter = missing_data_prop(combat_nofilter_expr, sort=sort)
# na_nocombat_nofilter = missing_data_prop(nocombat_nofilter_expr, sort=sort)
# na_microarray_logfc = missing_data_prop(microarray_logfc, sort=sort)
# na_tcga_logfc = missing_data_prop(tcga_logfc, sort=sort)
# na_tcga_expr = missing_data_prop(tcga_expr, sort=sort)
# na_combined_rank = missing_data_prop(combined_rank, sort=sort)
na_clinical_info = missing_data_prop(clinical_info)
# NOTE: Only clinical information contains missing values

# Counts by dataset
dataset_counts = group_counts(clinical_info, col="Dataset",
                              return_counts=True)
status_counts = group_counts(clinical_info, col="Disease Status",
                             return_counts=True)
histology_counts = group_counts(clinical_info, col="Histology",
                                return_counts=True)
gender_counts = group_counts(clinical_info, col="Gender",
                             return_counts=True)
stage_counts = group_counts(clinical_info, col="Stage",
                            return_counts=True)
surv_counts = group_counts(clinical_info, col="Survival",
                           return_counts=True)
smoking_count = group_counts(clinical_info, col="Smoking",
                             return_counts=True)
# NOTE: Most data points are missing based on the study they were taken from


# Checking feature values
# -----------------------
# # Scan through non-numeric feature classes
# stage = clinical_info[~clinical_info["Stage"].isnull()]
# histology = clinical_info[~clinical_info["Histology"].isnull()]
# surv_month = clinical_info[~clinical_info["Survival Month"].isnull()]
# smoking = clinical_info[~clinical_info["Smoking"].isnull()]
# NOTE: "Stage", "Histology" and "Smoking" need to be recoded.
# "Survival Month" should be removed


# Check numeric feature classes
desc = clinical_info.describe()
# NOTE: No peculiar values


# Preliminary cleaning of data
# ----------------------------
df = clinical_info.copy()

# Recode "Stage"
df["Stage_opt"] = df["Stage"].apply(
    lambda x: "1B" if x==" pT2N0" else STAGE_MAP[x])
# Optimistic recoding of "Stage" (pT2N0 --> 1B)

df["Stage_pes"] = df["Stage"].apply(
    lambda x: "2A" if x==" pT2N0" else STAGE_MAP[x])
# Pessimistic recoding of "Stage" (pT2N0 --> 2A)


# Recode "Smoking"
df["Smoking_opt"] = df["Smoking"].apply(
    lambda x: "Never" if x=="Non-smoking" else SMOKING_MAP[x])
# Optimistic recoding of "Smoking" (non-smoking --> never)

df["Smoking_pes"] = df["Smoking"].apply(
    lambda x: "Former" if x=="Non-smoking" else SMOKING_MAP[x])
# Pessimistic recoding of "Smoking" (non-smoking --> former)


# Recode "Histology"
df["Histology"] = df["Histology"].map(HISTOLOGY_MAP)



# # Remove columns
# rm = ["Race",
#       "Recurrence",
#       "Others",
#       "TNM stage (T)",
#       "TNM stage (N)",
#       "TNM stage (M)"]


# =============================================================================
# EXPLORATORY PLOTS
# =============================================================================

figpath = os.path.join(ROOT, "figures", "exploratory")
create_directory(figpath)


# Plot age distributions
# ----------------------
hues = ["Dataset", "Disease Status", "Gender", "Histology"]

for x, hue in [("Age", hue) for hue in hues]:
    title = x+" by "+hue.lower()
    
    fig, ax = plt.subplots(figsize=(8,8))
    sns.kdeplot(data=df, x=x, hue=hue, ax=ax)
    # fig.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    ax.set_title(title)
    fig.tight_layout()
    
    name = title.lower().replace(" ", "_")
    path = os.path.join(figpath, name+".png")
    fig.savefig(path, bbox_inches="tight")
    

# "Optimistic" vs. "Pessimistic" estimates
subhues = ["Stage", "Smoking"]
for x, hue in [("Age", hue) for hue in subhues]:
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,8))
    title = x+" by "+hue.lower()
    fig.suptitle(title)
    for i, est in enumerate(["Optimistic", "Pessimistic"]):
        title = x+" by "+hue.lower()+" ("+est+")"
        subhue = hue+"_"+est.lower()[0:3]
        
        sns.kdeplot(data=df, x=x, hue=subhue, ax=ax[i])
        # fig.legend(bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
        ax[i].set_title(est)
        fig.tight_layout()
    
        name = x+" by "+subhue
        path = os.path.join(figpath, name.lower().replace(" ", "_")+".png")
        fig.savefig(path, bbox_inches="tight")