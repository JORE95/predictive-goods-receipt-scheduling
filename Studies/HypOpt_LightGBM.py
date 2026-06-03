
import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))

import pandas as pd
import sqlalchemy as sa
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import optuna
import sqlalchemy as sa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import Get_Data as gd


Model="LightGBM"
path="Database/DB_params.db"
engine = sa.create_engine("sqlite:///" + path)
engine.connect()


# %%
def Run_Optuna(runs=100, folds=5, Train_data=None):
    X = Train_data.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y = Train_data["verzoegerung_bin_5"]
    y = y.astype(int)
    w = Train_data["w_time"]
    w=w.astype(float)

    fixed_params={"objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "verbosity": -1}

    def objective(trial, X=X, y=y, w=w,fixed_params=fixed_params):
        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        encoder = trial.suggest_categorical("encoder", ["ordinal", "onehot"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])

        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "goss"])

        params = {
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "boosting_type": boosting_type,
    
        }

        if boosting_type == "gbdt":
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        elif boosting_type == "goss":
            params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.3)
            params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.2)

        params.update(fixed_params)

        model = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
            X_tr, X_va, y_tr, y_va, w_tr, w_va = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer, feature_output="LightGBM")
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)])
            pred_labels = model.predict(X_va)
            acc=accuracy_score(y_va, pred_labels, sample_weight=w_va)
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
    best_params.update(fixed_params)

    best_params_df = pd.DataFrame([best_params])
    best_params_df["model_name"] = Model
    best_params_df["best_value"] = trial.value
    best_params_df["saved_at"] = pd.Timestamp.now()
    best_params_df = best_params_df.sort_values(by="best_value", ascending=False)
    best_params_df.to_sql(Model, con=engine, if_exists="replace", index=False)

if __name__ == "__main__":
    Run_Optuna()




