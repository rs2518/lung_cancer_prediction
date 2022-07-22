import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from lcml import ROOT
from lcml.utils import (load_clinical_info,
                        load_expression_data,
                        get_de_genes)



# =============================================================================
# DIMENSION REDUCTION
# =============================================================================

# Load data
# ---------
clinical_info = load_clinical_info()
expr = load_expression_data(source="microarray", corrected=True,
                            filtered=True)
X_full = expr.loc[clinical_info.index, :]
X = X_full.loc[:, get_de_genes()]

# Transform data with PCA
pca_full = PCA()
X_full = pca_full.fit_transform(X_full)

pca = PCA()
X = pca.fit_transform(X)


# Pie chart for explained variance (all genes vs. DE genes only)
# --------------------------------------------------------------
total_variance_full = sum(pca_full.explained_variance_)
total_variance = sum(pca.explained_variance_)
proportions = [total_variance/total_variance_full,
               1 - total_variance/total_variance_full]
labels = ["DE genes (p=%s)" % (X.shape[1]),
          "Non-DE genes (p=%s)" % (expr.shape[1]-X.shape[1])]

fig, ax = plt.subplots()
ax.pie(proportions, labels=labels, autopct='%1.1f%%')
ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle("Proportion of total explained variance")
plt.tight_layout()

# NOTE: 1% of the features (DE genes) account for 5.6% of the total variability
# in the dataset


# Plot cumulative variance
# ------------------------
def scree_plot(explained_variance, n_components=None, max_variance=1., 
               title=None, bar_kwargs={}, line_kwargs={}):
    """Return scree plot given explained variance ratios from PCA
    
    Directly limit the number of components displayed with 'n_components' or
    limit the number of components based on the maximum cumulative variance
    using 'max_variance' (set n_components=None if using max_variance).
    """
    # Default settings
    if title is None:
        title = ""
    if bar_kwargs == {}:
        bar_kwargs = dict(color=sns.color_palette("crest_r", n_components))
    if line_kwargs == {}:
        line_kwargs = dict(marker="D", color="r")
        
    if n_components is None:
        n_components =\
            len(explained_variance[explained_variance<=max_variance])
    
    x = np.arange(n_components)+1
    var = explained_variance[0:n_components]
    
    fig, ax = plt.subplots(figsize=(15,6))
    ax.bar(x=np.arange(n_components)+1, height=var, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xlabel("Component")
    ax.set_ylabel("Proportion of explained variance", fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(var), **line_kwargs)
    ax2.set_ylabel("Cumulative explained variance", fontsize=14)
    ax.set_title(title, fontsize=17)
    
    plt.tight_layout()
    
    return fig

fig1 = scree_plot(pca_full.explained_variance_ratio_, 
                  n_components=20,
                  title="Scree plot (all genes)")

fig2 = scree_plot(pca.explained_variance_ratio_,
                  n_components=20,
                  title="Scree plot (DE genes only)")

## TODO: SAVE PLOTS