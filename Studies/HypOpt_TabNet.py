# %%
import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import Get_Data as gd
import torch
import pandas as pd
import torch
import sqlalchemy as sa
from sklearn.model_selection import StratifiedKFold

Model="TabNet"
path="Database/DB_params.db"
engine = sa.create_engine("sqlite:///" + path)
engine.connect()

# %%



# %%
def Run_Optuna(runs=100, epochs=100, folds=5, Train_data=None):
    
    X = Train_data.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y = Train_data["verzoegerung_bin_5"]
    y = y.astype(int)
    w = Train_data["w_time"]
    w=w.astype(float)


    def objective(trial, X=X, y=y, w=w):

        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        encoder = trial.suggest_categorical("encoder", ["ordinal", "onehot"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])

        ## Modelspezifische Hyperparameter
        n_d = trial.suggest_categorical("n_d", [16, 32, 64])
        lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)
        params = {
            "n_d": n_d,
            "n_a": n_d,
            "n_steps": trial.suggest_int("n_steps", 6, 10),
            "gamma": trial.suggest_categorical("gamma", [1.0, 1.2, 1.5]),
            "lambda_sparse": trial.suggest_categorical("lambda_sparse", [0.0, 1e-6, 1e-5, 1e-4]),
            "momentum": trial.suggest_categorical("momentum", [0.02, 0.05, 0.1, 0.2]),
            "optimizer_params": dict(
                lr=lr),
            "scheduler_params": {
                "step_size": trial.suggest_categorical("step_size", [10, 20]),
                "gamma": trial.suggest_categorical("scheduler_gamma", [0.5, 0.8, 0.9])
            }
                }

        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2028 ])
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []
        

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
            X_tr, X_va, y_tr, y_va, w_tr, w_va, cat_idxs, cat_dims = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer, feature_output="TabNet")

            clf = TabNetClassifier(
                **params,
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                seed=42,
            )

            clf.fit(
                X_train=X_tr,
                y_train=y_tr,
                weights=w_tr,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                eval_set=[(X_va, y_va)],
                eval_weights=[w_va],
                eval_metric=['accuracy'],   
                max_epochs=epochs,
                patience=50,
                batch_size=batch_size,
                verbose=0
            )


            y_pred = clf.predict(X_va)
            acc=accuracy_score(y_va, y_pred, sample_weight=w_va)
            scores.append(acc)
            trial.report(acc, step=i)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)
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




