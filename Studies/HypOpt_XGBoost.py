
import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))


import pandas as pd
import sqlalchemy as sa
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
import sqlalchemy as sa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import Get_Data as gd
Model="XGBoost"


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
    print("y unique:", np.unique(y))

    fixed_params = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "num_class": 5,
        "booster": "gbtree",
        "tree_method": "hist",    
        "predictor": "gpu_predictor",
        "eval_metric": "mlogloss"
    }


    def objective(trial, X=X, y=y, w=w,  fixed_params=fixed_params):

        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        encoder = trial.suggest_categorical("encoder", ["onehot", "ordinal"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])


        params = {
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),

        }

        number_ouf_boost_rounds = trial.suggest_int("num_boost_round", 50, 300) 

        params.update(fixed_params)



        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
            X_tr, X_va, y_tr, y_va, w_tr, w_va = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer)
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
            dval   = xgb.DMatrix(X_va, label=y_va, weight=w_va)
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=number_ouf_boost_rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            pred_labels = bst.predict(dval, output_margin=False).argmax(axis=1)
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