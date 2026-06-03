
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

def calc_metrics(y_true, y_pred, sample_weight=None, labels=None):
    metrics = {}

    metrics["accuracy_weighted"] = accuracy_score(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )

    metrics["mean_absolute_error_weighted"] = mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )

    metrics["f1_weighted"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )


    metrics["precision_weighted"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )

    metrics["recall_weighted"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )

    return metrics




def calc_metrics(y_true, y_pred, sample_weight=None, labels=None):
    metrics = {}

    metrics["accuracy_weighted"] = accuracy_score(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )

    metrics["mean_absolute_error_weighted"] = mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )

    metrics["f1_weighted"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )


    metrics["precision_weighted"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )

    metrics["recall_weighted"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=sample_weight,
        zero_division=0
    )

    return metrics



class ConfidenceIntervalCalculator:
    def __init__(self):
        self.bootstrap_results = pd.DataFrame(columns=[
            "Model",
            "metric",
            "mean_bootstrap",
            "std_bootstrap",
            "ci_lower_95",
            "ci_upper_95"
        ])

        self.bootstrap_samples = pd.DataFrame()
        self.ci_df = None

    def bootstrap_confidence_intervals(
        self,
        model_name,
        y_true,
        y_pred,
        sample_weight=None,
        n_bootstrap=2000,
        alpha=0.05,
        random_state=42):

        rng = np.random.default_rng(random_state)
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
    

    
        sample_weight = np.asarray(sample_weight).astype(float)

        n = len(y_true)
        bootstrap_rows = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)

            y_true_b = y_true[idx]
            y_pred_b = y_pred[idx]
            w_b = sample_weight[idx]

            metrics_b = calc_metrics(
                y_true_b,
                y_pred_b,
                sample_weight=w_b,
            )

            bootstrap_rows.append(metrics_b)

        bootstrap_df = pd.DataFrame(bootstrap_rows)
        bootstrap_df["Model"] = model_name

        ci_rows = []

        for metric in bootstrap_df.drop(columns=["Model"]).columns:
            values = bootstrap_df[metric].dropna()
            ci_rows.append({
                "Model": model_name,
                "metric": metric,
                "mean_bootstrap": values.mean(),
                "std_bootstrap": values.std(),
                "ci_lower_95": values.quantile(alpha / 2),
                "ci_upper_95": values.quantile(1 - alpha / 2)
            })

        ci_df = pd.DataFrame(ci_rows).round(4)

        self.bootstrap_results = pd.concat(
            [self.bootstrap_results, ci_df],
            ignore_index=True
        )

        self.bootstrap_samples = pd.concat(
            [self.bootstrap_samples, bootstrap_df],
            ignore_index=True
        )

        self.ci_df = ci_df

    