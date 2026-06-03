
import numpy as np
import sqlalchemy as sa
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
    OrdinalEncoder,
)
import pandas as pd
import re




# %%
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cpu=torch.device("cpu")
print("device:", device)
if device.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
    torch.cuda.synchronize()


def clean_feature_names(names):
    cleaned = []
    for name in names:
        name = name.replace(",", "_")
        name = name.replace(" ", "_")

        name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
        
        cleaned.append(name)
    return cleaned



class CosineEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period=12):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X).astype(float)
        X = np.nan_to_num(X, nan=0)
        sin_component = np.sin(2 * np.pi * X / self.period)
        cos_component = np.cos(2 * np.pi * X / self.period)
        return np.column_stack((sin_component, cos_component))
    
    def get_feature_names_out(self, input_features=None):
        return [f"{input_features[0]}_sin", f"{input_features[0]}_cos"]
    


def Weight_transformer(w):
        w = w / w.mean()
        w = np.log1p(w)
        w = np.clip(w, 0.01, 2)
        return w

def preprocess_split(
    X_train,
    X_val,
    y_train,
    y_val,
    w_train,
    w_val,
    scaler="minmax",
    encoder="onehot",
    periodic_transformer=True,
    feature_output=False
):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    
    if periodic_transformer==True:
        num_cols = [c for c in num_cols
        if c not in ["w_time", "verzoegerung_bin_5", "Bestätigter_Liefermonat"]
        ]
    else:
        num_cols = [c for c in num_cols
        if c not in ["w_time", "verzoegerung_bin_5"]
        ]

    if scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "power":
        scaler_obj = PowerTransformer(method="yeo-johnson", standardize=True)
    elif scaler == "none":
        scaler_obj = "passthrough"
    else:
        raise ValueError("Invalid scaler choice.")

    num_pipe = Pipeline([("pt", scaler_obj),])

    if encoder == "onehot":
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoder == "ordinal":
        cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    elif encoder == "none":
        cat_transformer = "passthrough"
    else:
        raise ValueError("Invalid encoder choice.")
    
    cat_transformer = Pipeline([("encoder", cat_transformer),])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_transformer, cat_cols)

    ]

    if periodic_transformer:
        transformers.append(("cosine", CosineEncoder(), ["Bestätigter_Liefermonat"]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )



    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    w_train = Weight_transformer(w_train)
    w_val = Weight_transformer(w_val)



    if feature_output=="TabNet":
        train_df = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
        train_df["target"] = y_train.to_numpy()
        val_df = pd.DataFrame(X_val, columns=preprocessor.get_feature_names_out())
        val_df["target"] = y_val.to_numpy()
        cat_features = [col for col in train_df.columns if "cat__" in col or "remainder__" in col]
        cat_idxs = [train_df.columns.get_loc(col) for col in cat_features]
        cat_dims = [train_df[col].nunique() for col in cat_features]
        X_train = train_df.drop(columns=["target"]).to_numpy()
        X_val = val_df.drop(columns=["target"]).to_numpy()
        return X_train, X_val, y_train.to_numpy(), y_val.to_numpy(), w_train.to_numpy(), w_val.to_numpy(), cat_idxs, cat_dims
    

    if feature_output=="LightGBM":
            feature_names = preprocessor.get_feature_names_out()
            feature_names_clean = clean_feature_names(feature_names)
            X_train = pd.DataFrame(X_train, columns=feature_names_clean)
            X_val = pd.DataFrame(X_val, columns=feature_names_clean)
            return X_train,X_val, y_train.to_numpy(),y_val.to_numpy(),w_train.to_numpy(), w_val.to_numpy()
 
    
    if feature_output=="MLP" or feature_output=="Gandalf":
        X_train = torch.tensor(X_train.astype(np.float32), dtype=torch.float32)
        X_val = torch.tensor(X_val.astype(np.float32), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy().astype(int), dtype=torch.long)
        y_val= torch.tensor(y_val.to_numpy().astype(int), dtype=torch.long)
        w_train = torch.tensor(w_train.to_numpy().astype(np.float32), dtype=torch.float32)
        w_val   = torch.tensor(w_val.to_numpy().astype(np.float32), dtype=torch.float32)
        return X_train, X_val, y_train, y_val, w_train, w_val
    
    if feature_output=="CatBoost":
        feature_names = preprocessor.get_feature_names_out()
        feature_names_clean = clean_feature_names(feature_names)
        X_train = pd.DataFrame(X_train, columns=feature_names_clean)
        X_val = pd.DataFrame(X_val, columns=feature_names_clean)
        cat_features = [i for i, col in enumerate(X_train.columns)if "remainder__" in col]
        return X_train, X_val, y_train.to_numpy(), y_val.to_numpy(), w_train.to_numpy(), w_val.to_numpy(),cat_features
                
    else:
        return X_train, X_val, y_train.to_numpy(), y_val.to_numpy(), w_train.to_numpy(), w_val.to_numpy()

if __name__ == "__main__":
    preprocess_split()
