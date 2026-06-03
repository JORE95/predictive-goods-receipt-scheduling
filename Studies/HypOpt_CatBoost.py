
import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()
sys.path.append(str(BASE_PATH/"Funcitons"))

import pandas as pd
import sqlalchemy as sa
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import optuna
import sqlalchemy as sa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import Get_Data as gd
import torch



Model="CatBoost"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = CatBoostClassifier(task_type="GPU", devices="0")


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

    # %%
    fixed_params = {

            'one_hot_max_size': 25,      
            'max_ctr_complexity': 4,     #
      
            "task_type":"GPU", 
            "devices":"0",         
            'thread_count': -1,          
            'bootstrap_type': 'Bayesian', 
 
            'early_stopping_rounds': 50,
            'od_type': 'Iter',         
            'loss_function': 'MultiClass',  
            'eval_metric': 'Accuracy',      
            'random_strength': 1,       
            'bagging_temperature': 1,   
            
    }

    def objective(trial, X=X, y=y, w=w, fixed_params=fixed_params, folds=folds):

        scaler = trial.suggest_categorical("scaler", ["minmax", "standard", "power"])
        periodic_transformer = trial.suggest_categorical("periodic_transformer", [True, False])
        encoder = trial.suggest_categorical("encoder", ["ordinal", "onehot"])
        params = {
            'iterations': trial.suggest_int('iterations', 100, 150),          
            'learning_rate': trial.suggest_float("learning_rate",0.01, 0.3),    
            'depth': trial.suggest_int("depth", 4, 10),
            'l2_leaf_reg': trial.suggest_int("l2_leaf_reg", 1, 10),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 50),      

        }

        params.update(fixed_params)
        model = CatBoostClassifier(**params)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_va = w.iloc[train_idx], w.iloc[val_idx]
            X_tr, X_va, y_tr, y_va, w_tr, w_va, Categorical = gd.preprocess_split(X_tr_raw, X_va_raw, y_tr, y_va, w_tr, w_va, scaler=scaler, encoder=encoder, periodic_transformer=periodic_transformer, feature_output="CatBoost")
            params['cat_features'] = Categorical
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            pred_labels = model.predict(X_va)
            acc=accuracy_score(y_va, pred_labels, sample_weight=w_va)
            print(f"Fold {i+1} Accuracy: {acc}")
            scores.append(acc)
            trial.report(acc, step=i)

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
    best_params.update(fixed_params)
    best_params_df = pd.DataFrame([best_params])
    best_params_df["model_name"] = Model
    best_params_df["best_value"] = trial.value
    best_params_df["saved_at"] = pd.Timestamp.now()
    best_params_df = best_params_df.sort_values(by="best_value", ascending=False)
    best_params_df.to_sql(Model, con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    Run_Optuna()




