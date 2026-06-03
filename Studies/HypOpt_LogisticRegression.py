import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))

import numpy as np
import numpy as np
import optuna
import Get_Data as gd
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import Get_Data as gd

Model="Logistic Regression"

path="Database/DB_params.db"
engine = sa.create_engine("sqlite:///" + path)
engine.connect()

def Run_Optuna(runs=100, folds=5, Train_data=None):

    X = Train_data.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y = Train_data["verzoegerung_bin_5"]
    y = y.astype(int)
    w = Train_data["w_time"]
    w=w.astype(float)
  



    def objective(trial, X=X, y=y, w=w):

        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        encoder = trial.suggest_categorical("encoder", ["ordinal", "onehot"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])
        solver = trial.suggest_categorical(
            "solver",
            ["lbfgs","newton-cg", "newton-cholesky"]
        )

        C = trial.suggest_float("C", 1e-3, 100, log=True)


        max_iter = trial.suggest_int("max_iter", 2000, 3000)


        class_weight = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        )

        model = LogisticRegression(
            solver=solver,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        )



        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
            X_tr, X_va, y_tr, y_va, w_tr, w_va = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer)

            model.fit(X_tr, y_tr, sample_weight=w_tr)
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
 
    best_params_df = pd.DataFrame([best_params])
    best_params_df["model_name"] = Model
    best_params_df["best_value"] = trial.value
    best_params_df["saved_at"] = pd.Timestamp.now()
    best_params_df = best_params_df.sort_values(by="best_value", ascending=False)
    best_params_df.to_sql(Model, con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    Run_Optuna()  