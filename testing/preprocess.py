import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, ColumnTransformer


from lcml import load_data

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
df["Disease Status"] = df["Disease Status"].map({"Tumour":1, "Non-tumour":0})


# Preprocess features
# -------------------
X = df.drop(columns="Disease Status")
Y = df["Disease Status"]

categorical_columns = make_column_selector(dtype_include="category")(X)
numerical_columns = make_column_selector(dtype_include=np.number)(X)
oh = OneHotEncoder()
sc = StandardScaler()
preprocessor = ColumnTransformer([
    ("one-hot-encoder", oh, categorical_columns),
    ("standard_scaler", sc, numerical_columns)],
    verbose_feature_names_out=False)

X = preprocessor.fit_transform(X)
X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())