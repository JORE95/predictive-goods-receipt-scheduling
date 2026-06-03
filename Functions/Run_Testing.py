# %%

import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()

sys.path.append(str(BASE_PATH))
sys.path.append(str(BASE_PATH / "Costum_Models"))
import Gandalf as Gan_model
import MLP as MLP_model
import Confinv
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
import pytorch_tabnet
import pandas as pd
import numpy as np
import torch.nn as nn

import sqlalchemy as sa
import Results as res
import Get_Data as gd
import time
from sklearn.model_selection import train_test_split

import torch

from torch.utils.data import DataLoader
import torch.utils.data as data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu=torch.device("cpu")
import pytorch_tabnet

print("Python:", sys.executable)
print("pytorch_tabnet path:", pytorch_tabnet.__file__)

# %% Test Models

def test_models():
#%% Load Data and Database
    path="Database/DB_params.db"
    engine = sa.create_engine("sqlite:///" + path)
    engine.connect()


    path_train="Training_Test_Datensatz/Training_Datensatz.xlsx"
    path_test="Training_Test_Datensatz/Test_Datensatz.xlsx"


    Train= pd.read_excel(path_train)
    Test= pd.read_excel(path_test)

    X_test_raw = Test.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y_test_raw = Test["verzoegerung_bin_5"]
    w_test_raw = Test["w_time"]

    X_train_raw = Train.drop(columns=["verzoegerung_bin_5","w_time","Verzoegerungstage"])
    y_train_raw = Train["verzoegerung_bin_5"]
    w_train_raw = Train["w_time"]


    # %% Initialize Confidence Interval Calculator
    conf=Confinv.ConfidenceIntervalCalculator()

    # %% Logistic Regression


    Model="Logistic Regression"

    best_params=pd.read_sql("Logistic Regression", engine).iloc[0].to_dict()


    if best_params["class_weight"]=="balanced":
        best_params["class_weight"] = "balanced"
    else:
        best_params["class_weight"] = None

    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw,scaler=best_params.pop("scaler"), encoder=best_params.pop("encoder") , periodic_transformer=best_params.pop("periodic_transformer"))
    model = LogisticRegression(
        solver=best_params["solver"],
        C=best_params["C"],
        max_iter=best_params["max_iter"],
        class_weight=best_params["class_weight"],
        random_state=42
    )



    start=time.time()
    model.fit(X_train, y_train, sample_weight=w_train)
    end=time.time()


    X_test = np.nan_to_num(X_test, nan=0)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    traing_time=end-start

    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )

    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)

    # %% CatBoost

    Model="CatBoost"
    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()
    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, scaler=best_params.pop("scaler"), encoder=best_params.pop("encoder") , periodic_transformer=best_params.pop("periodic_transformer"))
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)
    best_params.pop("model_name")
    best_params.pop("best_value")
    best_params.pop("saved_at")

    final_model = CatBoostClassifier(**best_params, allow_writing_files=False)

    start=time.time()
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        early_stopping_rounds=50,
        verbose=False
    )
    end=time.time()
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)
    traing_time=end-start

    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )
    res.save_frames(Model, y_pred.flatten(), y_proba ,y_test, w_test, training_time=traing_time)

    # %% Gandalf

    Model="Gandalf"
    epochs=100
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()
    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, scaler=best_params.pop("scaler"), encoder=best_params.pop("encoder") , periodic_transformer=best_params.pop("periodic_transformer") , feature_output="MLP")
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)


    best_params.pop("model_name")
    best_params.pop("best_value")
    best_params.pop("saved_at")

    batch_size=best_params.pop("batch_size")

    train_dataset = data_utils.TensorDataset(X_tr, y_tr, w_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = data_utils.TensorDataset(X_val, y_val, w_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Gan_model.Model(
                        input_dim=X_train.shape[1],
                        gflu_stages=best_params["gflu_stages"],
                        gflu_dropout=best_params["gflu_dropout"],
                        gflu_feature_init_sparsity=best_params["gflu_feature_init_sparsity"],
                        learnable_sparsity=True,
                        )

    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=best_params.pop("weight_decay"))
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, min_lr=1e-6)


    start=time.time()
    accuracay_plot=model.fit(train_dataloader, val_dataloader,epochs, optimizer, sheduler, loss_fn)
    end=time.time()

    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)


    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )

    traing_time=end-start
    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)

    # %% LightGBM
    Model="LightGBM"

    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()

    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw,scaler=best_params.pop("scaler"), encoder=best_params.pop("encoder") , periodic_transformer=best_params.pop("periodic_transformer"))
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)
    best_params.pop("model_name")
    best_params.pop("best_value")
    best_params.pop("saved_at")

    model = LGBMClassifier(**best_params)

    start=time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        sample_weight=w_tr,
        eval_sample_weight=[w_val]
    )
    end=time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    traing_time=end-start
    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )
    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)

    # %% Multilayer Perceptron

    Model="Multilayer Perceptron"
    epochs=100
    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()
    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, scaler=best_params.pop("scaler"), encoder=best_params.pop("encoder") , periodic_transformer=best_params.pop("periodic_transformer") , feature_output="MLP")
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)

    best_params.pop("model_name")
    best_params.pop("best_value")
    best_params.pop("saved_at")

    batch_size=best_params.pop("batch_size")

    train_dataset = data_utils.TensorDataset(X_tr, y_tr, w_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = data_utils.TensorDataset(X_val, y_val, w_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    model=MLP_model.Model(input_dim=X_train.shape[1])

    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=best_params.pop("weight_decay"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, min_lr=1e-6)


    start=time.time()
    accuracay_plot=model.fit(train_dataloader, val_dataloader,epochs, optimizer, scheduler)
    end=time.time()

    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)

    traing_time=end-start

    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )
    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)

    # %% TabNet
    Model="TabNet"
    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()

    X_train, X_test, y_train, y_test, w_train, w_test, cat_idxs, cat_dims = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw,scaler="none", encoder="ordinal", periodic_transformer=False, feature_output="TabNet")
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)

    scaler = best_params.pop("scaler")
    encoder = best_params.pop("encoder")
    periodic_transformer = best_params.pop("periodic_transformer")
    batch_size = best_params.pop("batch_size")
    optimizer_params=best_params.pop("lr")
    step_size=best_params.pop("step_size")
    scheduler_gamma=best_params.pop("scheduler_gamma")

    best_params.pop("model_name")
    best_params.pop("best_value")
    best_params.pop("saved_at")



    model = TabNetClassifier(
        **best_params,
        optimizer_params=dict(lr=optimizer_params),
        scheduler_params=dict(step_size=step_size, gamma=scheduler_gamma),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42,
    )

    start=time.time()

    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        weights=w_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=['accuracy'],
        max_epochs=300,
        patience=100,
        batch_size=batch_size,
    )
    end=time.time()


    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)

    y_pred=model.predict(X_test)
    y_proba=model.predict_proba(X_test)

    traing_time=end-start

    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )
    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)


    # %% XGBoost

    Model="XGBoost"

    best_params=pd.read_sql(Model, engine).iloc[0].to_dict()

    scaler = best_params.pop("scaler")
    encoder = best_params.pop("encoder")
    periodic_transformer = best_params.pop("periodic_transformer")
    number_ouf_boost_rounds = best_params.pop("num_boost_round")

    X_train, X_test, y_train, y_test, w_train, w_test = gd.preprocess_split(X_train_raw, X_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw,scaler=scaler,encoder=encoder,periodic_transformer=periodic_transformer)
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split( X_train, y_train, w_train,test_size=0.2,stratify=y_train,random_state=42)



    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dval   = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    dtest  = xgb.DMatrix(X_test, label=y_test, weight=w_test)

    start=time.time()
    bst = xgb.train(
        best_params,
        dtrain,
        num_boost_round=number_ouf_boost_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    end=time.time()
    traing_time=end-start
    y_proba = bst.predict(dtest, output_margin=False)
    y_pred = y_proba.argmax(axis=1)
    end=time.time()


    traing_time=end-start

    conf.bootstrap_confidence_intervals(
        model_name=Model,
        y_true=y_test,
        y_pred=y_pred,
        sample_weight=w_test
    )

    res.save_frames(Model, y_pred, y_proba ,y_test, w_test, training_time=traing_time)


    # %% Save Confidence Intervals to Database
    sqlite_path = "Database/DB_results.db"
    engine = sa.create_engine("sqlite:///" + sqlite_path)
    engine.connect()
    df=conf.bootstrap_results.round(4)
    df.to_sql("ConfidenceIntervals", con=engine, if_exists="replace", index=False)


if __name__ == "__main__":
    test_models()



