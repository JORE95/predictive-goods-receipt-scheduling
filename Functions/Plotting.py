# %%
import pandas as pd
import sqlalchemy as sa
import numpy as np  

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#
# %%
sqlite_path = "Database/DB_results.db"
engine = sa.create_engine("sqlite:///" + sqlite_path)
engine.connect()


def plot_confidence_intervalls():

    df = pd.read_sql("SELECT * FROM ConfidenceIntervals", con=engine)

    num_cols = ["mean_bootstrap", "ci_lower_95", "ci_upper_95"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    metrics = {
        "accuracy_weighted": "Accuracy",
        "f1_weighted": "F1-Score",
        "MAE": "MAE"
    }


    metrics_to_plot = list(metrics.items())

    order_df = df[df["metric"] == "accuracy_weighted"].copy()
    order_df = order_df.sort_values("mean_bootstrap", ascending=True)
    model_order = order_df["Model"].tolist()

    fig = make_subplots(
        rows=1,
        cols=len(metrics_to_plot),
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=[label for _, label in metrics_to_plot]
    )

    for col_idx, (metric, xlabel) in enumerate(metrics_to_plot, start=1):

        plot_df = df[df["metric"] == metric].copy()

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=model_order,
            ordered=True
        )

        plot_df = plot_df.sort_values("Model").reset_index(drop=True)

        xerr_lower = plot_df["mean_bootstrap"] - plot_df["ci_lower_95"]
        xerr_upper = plot_df["ci_upper_95"] - plot_df["mean_bootstrap"]

        x_min = plot_df["ci_lower_95"].min() - 0.02
        x_max = plot_df["ci_upper_95"].max() + 0.12

        for _, row in plot_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[x_min, row["mean_bootstrap"]],
                    y=[row["Model"], row["Model"]],
                    mode="lines",
                    line=dict(
                        color="lightgray",
                        width=2
                    ),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1,
                col=col_idx
            )

        fig.add_trace(
            go.Scatter(
                x=plot_df["mean_bootstrap"],
                y=plot_df["Model"],
                mode="markers",
                marker=dict(size=8),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=xerr_upper,
                    arrayminus=xerr_lower,
                    thickness=1.5,
                    width=4
                ),
                customdata=plot_df[["ci_lower_95", "ci_upper_95"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{xlabel}: %{{x:.3f}}<br>"
                    "95%-CI: [%{customdata[0]:.3f}; %{customdata[1]:.3f}]"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=1,
            col=col_idx
        )

        for _, row in plot_df.iterrows():

            label = (
                f'{row["mean_bootstrap"]:.3f} '
                f'[{row["ci_lower_95"]:.3f}; {row["ci_upper_95"]:.3f}]'
            )

            if row["Model"] == "LightGBM":
                label = f"<b>{label}</b>"

            fig.add_annotation(
                x=row["ci_upper_95"] + 0.004,
                y=row["Model"],
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12),
                row=1,
                col=col_idx
            )

        fig.update_xaxes(
            title_text=xlabel,
            range=[x_min, x_max],
            showgrid=True,
            griddash="dash",
            gridcolor="rgba(0,0,0,0.25)",
            zeroline=False,
            row=1,
            col=col_idx
        )

        fig.update_yaxes(
            categoryorder="array",
            categoryarray=model_order,
            row=1,
            col=col_idx
        )

    fig.update_yaxes(
        title_text="Modellpipeline",
        row=1,
        col=1
    )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=450,
        margin=dict(l=120, r=40, t=60, b=60),
        font=dict(size=12)
    )

    fig.show(renderer="browser")





# %%

def overview_plotly():

    df = pd.read_sql("SELECT * FROM model_results", con=engine)

    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

    df = df[df["timestamp"].astype(str).str.contains("2026-06-03", na=False)]

    df["rank"] = df.groupby(df.columns[0])["accuracy"].rank(
        ascending=False,
        method="first"
    )

    df = df.sort_values(by=["accuracy"], ascending=False)
    df = df.replace("None", np.nan).fillna(0)
    df = df[df["rank"] == 1.0]

    df = df.drop(columns=["rank", "timestamp", "rank_overall"])

    df.columns = [
        "Model",
        "Accuracy",
        "MAE",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1-Score",
        "Training Time (s)"
    ]

    df = df.reset_index(drop=True)

    display_df = df.copy()

    for col in display_df.columns:
        if col != "Model":
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            )

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[220, 150, 150, 150, 150, 150, 150, 180],
                header=dict(
                    values=[f"<b>{col}</b>" for col in display_df.columns],
                    font=dict(size=25),
                    align=["left"] + ["center"] * (len(display_df.columns) - 1),
                    height=45
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    font=dict(size=25),
                    align=["left"] + ["center"] * (len(display_df.columns) - 1),
                    height=42
                )
            )
        ]
    )

    fig.update_layout(
        width=1400,
        height=max(350, 120 + len(display_df) * 45),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.show(renderer="browser")
  






