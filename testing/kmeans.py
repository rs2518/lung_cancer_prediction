import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lcml import ROOT, load_data
from lcml.utils import (load_clinical_info,
                        load_expression_data,
                        get_de_genes)



# =============================================================================
# CLUSTER ANALYSIS
# =============================================================================

# Load data
# ---------
clinical_info = load_clinical_info()
expr = load_expression_data(source="microarray", corrected=True,
                            filtered=True)
X_full = expr.loc[clinical_info.index, :]
X = X_full.loc[:, get_de_genes()]


# Elbow plots for original features
# ---------------------------------
n_clusters = 11    # One more than number of disease subtypes
subtitle = ["All genes (p=%s)" % (X_full.shape[1]),
            "DE genes only (p=%s)" % (X.shape[1])]
colours = ["r", "b"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
for i, data in enumerate([X_full, X]):
    inertia = []
    for k in range(1, n_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=220301)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    sns.lineplot(x=np.arange(n_clusters)+1, y=inertia, ax=axes[i],
                 marker="X", markerfacecolor=colours[i],
                 linestyle="--", color=colours[i])
    axes[i].set_xlabel("Number of clusters")
    axes[i].set_title(subtitle[i])
axes[0].set_ylabel("Within Cluster Sum of Squares (inertia)")

plt.suptitle("Elbow plots")
plt.tight_layout()

# NOTE: Elbow plot is more distinct when dataset is reduced to DE genes


# Elbow plots for transformed features
# ------------------------------------



# NOTE: Number of components derived from PCA analysis (pca.py)

## TODO: PERFORM ANALYSES FOR DIFFERENT PREPROCESSING STEPS