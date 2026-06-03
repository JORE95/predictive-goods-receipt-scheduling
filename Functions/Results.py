# %%
import sqlalchemy as sa
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,accuracy_score, mean_absolute_error

# %%
sqlite_path = "/mnt/c/Users/jrech/OneDrive - Modehaus und Trachtenhaus Rechenauer GmbH/Desktop/Privat/MBA Workflow/Masterarbeit/CODE_BASE/DATEN/DB_results_new.db"
engine = sa.create_engine("sqlite:///" + sqlite_path)
engine.connect()


# %%


def save_results(model_name, w_accuracy="None",mae="None", roc_auc="None", precision="None", recall="None", f1_score="None", training_time="None"):

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    new_result = pd.DataFrame([{
        "model_name": model_name,
        "accuracy": np.round(w_accuracy, 4) if w_accuracy != "None" else None,
        "mean_absolute_error": np.round(mae, 4) if mean_absolute_error != "None" else None,
        "roc_auc": np.round(roc_auc, 4) if roc_auc != "None" else None,
        "precision": np.round(precision, 4) if precision != "None" else None,
        "recall": np.round(recall, 4) if recall != "None" else None,
        "f1_score": np.round(f1_score, 4) if f1_score != "None" else None,
        "training_time": training_time,
        "timestamp": timestamp 
    }])
    
    if w_accuracy != "None" and w_accuracy <= 0.4:
        return
    new_result.to_sql("model_results", con=engine, if_exists="append", index=False)
    df=pd.read_sql("SELECT * FROM model_results", con=engine)
    df["rank"] = df.groupby(df.columns[0])["accuracy"].rank(ascending=False, method="first")
    df=df.sort_values(by=["accuracy"], ascending=False)
    df["rank_overall"] = df["accuracy"].rank(ascending=False, method="first")
    df.to_sql("model_results", con=engine, if_exists="replace", index=False)


def save_frames(Model, preds, y_proba, y_test, w_test, training_time="None"):

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame({
        "preds": preds,
        "y_test": y_test,
        "w_test": w_test,
        "timestamp": timestamp
    })

    accuracy=accuracy_score(df["y_test"], df["preds"], sample_weight=df["w_test"])
    precision=precision_score(df["y_test"], df["preds"], average="weighted", sample_weight=df["w_test"], zero_division=0)
    recall=recall_score(df["y_test"], df["preds"], average="weighted", sample_weight=df["w_test"], zero_division=0)
    f1=f1_score(df["y_test"], df["preds"], average="weighted", sample_weight=df["w_test"], zero_division=0)
    MAE = mean_absolute_error(df["y_test"], df["preds"], sample_weight=df["w_test"])
    try:
         auc=roc_auc_score(df["y_test"], y_proba, sample_weight=df["w_test"], multi_class="ovr")

    except:
        print("Warning: Probabilities do not sum to 1. Adjusting values.")
        y_proba = np.nan_to_num(y_proba, nan=0.0)
        if np.abs(np.sum(y_proba, axis=1).max())>1.0:
                row_sums = y_proba.sum(axis=1, keepdims=True)
                y_proba=y_proba-((row_sums-1.00)/5)
                auc=roc_auc_score(df["y_test"], y_proba, sample_weight=df["w_test"], multi_class="ovr")
        else:
            auc=roc_auc_score(df["y_test"], y_proba, sample_weight=df["w_test"], multi_class="ovr")

    df.to_sql(Model, con=engine, if_exists="replace", index=False)
    training_time=np.round(training_time,2) if training_time != "None" else None
    save_results(Model, accuracy,MAE, auc, precision, recall, f1, training_time)
    print(f"Results for {Model} saved successfully at {timestamp}.")


if __name__ == "__main__":
        save_results()


