import numpy as np
import pandas as pd

from lcml import CLINICAL_INFO_COLS


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

# Missing clinical information
# ----------------------------
sort="descending"
# na_patient_status = missing_data_prop(patient_status, sort=sort)
# na_combat_filter = missing_data_prop(combat_filter_expr, sort=sort)
# na_combat_nofilter = missing_data_prop(combat_nofilter_expr, sort=sort)
# na_nocombat_nofilter = missing_data_prop(nocombat_nofilter_expr, sort=sort)
# na_microarray_logfc = missing_data_prop(microarray_logfc, sort=sort)
# na_tcga_logfc = missing_data_prop(tcga_logfc, sort=sort)
# na_tcga_expr = missing_data_prop(tcga_expr, sort=sort)
# na_combined_rank = missing_data_prop(combined_rank, sort=sort)
na_clinical_info = missing_data_prop(clinical_info, sort=sort)
# NOTE: Only clinical information contains missing values


# Counts by dataset (clinical information)
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