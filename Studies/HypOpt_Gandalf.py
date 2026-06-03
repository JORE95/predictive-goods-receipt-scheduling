import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))
sys.path.append(str(BASE_PATH / "Costum_Models"))

import Gandalf as Gan



import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import sqlalchemy as sa
import optuna


import Get_Data as gd
import Gandalf as Gan
Model="Gandalf"

# %%
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cpu=torch.device("cpu")
print("device:", device)
if device.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
    torch.cuda.synchronize()

path="Database/DB_params.db"
engine = sa.create_engine("sqlite:///" + path)
engine.connect()





# %%

def Run_Optuna(runs=20,epochs=100, folds=3, Train_data=None):
    X = Train_data.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y = Train_data["verzoegerung_bin_5"]
    y = y.astype(int)
    w = Train_data["w_time"]
    w=w.astype(float)
    
    def objective(trial, X=X, y=y, w=w):

        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        encoder = trial.suggest_categorical("encoder", ["onehot", "ordinal"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gflu_stages= trial.suggest_categorical("gflu_stages", [5, 10, 15])
        gflu_dropout = trial.suggest_categorical("gflu_dropout", [0.0, 0.1, 0.2])
        gflu_feature_init_sparsity = trial.suggest_categorical("gflu_feature_init_sparsity", [0.2, 0.4, 0.6])
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        scores = []
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
                X_train, X_val, y_train, y_val, w_train, w_val = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer, feature_output="MLP")
                train_dataset = data_utils.TensorDataset(X_train, y_train, w_train)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataset = data_utils.TensorDataset(X_val, y_val, w_val)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                model = Gan.Model(
                                    input_dim=X_train.shape[1],
                                    gflu_stages=gflu_stages,
                                    gflu_dropout=gflu_dropout,
                                    gflu_feature_init_sparsity=gflu_feature_init_sparsity,
                                    learnable_sparsity=True,
                                    )

                optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=weight_decay)
                sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, min_lr=1e-6)
                accuracay_plot=model.fit(train_dataloader, val_dataloader,epochs, optimizer, sheduler, loss_fn)
                
                scores.append(accuracay_plot)
                trial.report(accuracay_plot, step=i)

                if trial.should_prune():
                    raise optuna.TrialPruned()
                
        return np.mean(scores)

    # %%
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=runs, show_progress_bar=True)
    trial = study.best_trial

    print("Value:", trial.value)
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    best_params = dict(trial.params)

    best_params_df = pd.DataFrame([best_params])
    best_params_df["model_name"] = Model
    best_params_df["best_value"] = trial.value
    best_params_df["saved_at"] = pd.Timestamp.now()
    best_params_df = best_params_df.sort_values(by="best_value", ascending=False)
    best_params_df.to_sql(Model, con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    Run_Optuna()