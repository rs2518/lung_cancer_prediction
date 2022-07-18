import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from lcml import (TRAIN_TEST_PARAMS,
                  load_data)
from lcml.utils import feature_preprocessor

## ONLY USE AFTER THOROUGHLY CHECKING WARNINGS
# import warnings

# warnings.filterwarnings("ignore")



# =============================================================================
# PREPROCESS DATA
# =============================================================================

# Load data
# ---------
# Expression levels only, DE genes only
df = load_data(label="Disease Status", expr_only=True, top_genes=True)

X = df.drop(columns="Disease Status")
y = df["Disease Status"].map({"Tumour":1, "Non-tumour":0})


# Preprocess features
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)

preprocessor = feature_preprocessor(X)
X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                       columns=preprocessor.get_feature_names_out())

# preprocessor.named_transformers_["standard_scaler"].mean_