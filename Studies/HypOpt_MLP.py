import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))
sys.path.append(str(BASE_PATH/"Costum_Models"))


import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import sqlalchemy as sa
import optuna
import Get_Data as gd
import MLP as MLP_model

Model="Multilayer Perceptron"

# %%
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu=torch.device("cpu")


if device.type == "cuda":
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
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
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

                model=MLP_model.Model(input_dim=X_train.shape[1])

                optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, min_lr=1e-6)
                accuracay_plot=model.fit(train_dataloader, val_dataloader, epochs=epochs, optimizer=optimizer, scheduler=scheduler)


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

    best_params = dict(trial.params)

    best_params_df = pd.DataFrame([best_params])
    best_params_df["model_name"] = Model
    best_params_df["best_value"] = trial.value
    best_params_df["saved_at"] = pd.Timestamp.now()
    best_params_df = best_params_df.sort_values(by="best_value", ascending=False)
    best_params_df.to_sql(Model, con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    Run_Optuna()