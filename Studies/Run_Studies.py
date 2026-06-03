

import sys
from pathlib import Path
BASE_PATH = Path.cwd().absolute()

sys.path.append(str(BASE_PATH))
sys.path.append(str(BASE_PATH / "Studies"))
sys.path.append(str(BASE_PATH / "Functions"))

import HypOpt_CatBoost as Cat
import HypOpt_Gandalf as Gan
import HypOpt_LightGBM as LGBM
import HypOpt_LogisticRegression as Log
import HypOpt_MLP as mlp
import HypOpt_TabNet as Tab
import HypOpt_XGBoost as XGB

import pandas as pd
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu=torch.device("cpu")

path_train="Training_Test_Datensatz/Training_Datensatz.xlsx"

Train= pd.read_excel(path_train)

def Run_Studies_function():
    #Referenzmodel:
    Log.Run_Optuna(runs=100, folds=5, Train_data=Train)
    #Boositng Modelle:
    Cat.Run_Optuna(runs=100, folds=5, Train_data=Train)
    LGBM.Run_Optuna(runs=100, folds=5, Train_data=Train)
    XGB.Run_Optuna(runs=100,folds=5, Train_data=Train)

    #Deep Learning Modelle:
    Gan.Run_Optuna(runs=20,epochs=100, folds=3, Train_data=Train)
    mlp.Run_Optuna(runs=20,epochs=100, folds=3, Train_data=Train)
    Tab.Run_Optuna(runs=20,epochs=300, folds=3, Train_data=Train)



if __name__ == "__main__":
    Run_Studies_function()
