# %%
import pandas as pd
import sqlalchemy as sa
import numpy as np  

import matplotlib.pyplot as plt
#
# %%
sqlite_path = "Database/DB_results.db"
engine = sa.create_engine("sqlite:///" + sqlite_path)
engine.connect()



def confidence_intervalls():

    df=pd.read_sql("SELECT * FROM ConfidenceIntervals", con=engine)

    metrics = {
        "accuracy_weighted": "Accuracy",
        "f1_weighted": "F1-Score",
        "MAE": "MAE"
        
    }

    order_df = df[df["metric"] == "accuracy_weighted"].copy()
    order_df = order_df.sort_values("mean_bootstrap", ascending=True)
    model_order = order_df["Model"].tolist()


    y_pos = np.arange(len(model_order))



    fig, axes = plt.subplots(
        ncols=2,
        figsize=(12, 4.5),
        sharey=True
    )



    for ax, (metric, xlabel) in zip(axes, metrics.items()):
        print(f"Plotting metric: {metric}")
        print(f"Model order: {xlabel}")

        plot_df = df[df["metric"] == metric].copy()
        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=model_order,
            ordered=True
        )
        plot_df = plot_df.sort_values("Model").reset_index(drop=True)

        xerr_lower = plot_df["mean_bootstrap"] - plot_df["ci_lower_95"]
        xerr_upper = plot_df["ci_upper_95"] - plot_df["mean_bootstrap"]
        xerr = [xerr_lower, xerr_upper]
        

        ax.hlines(
            y=y_pos,
            xmin=plot_df["ci_lower_95"].min() - 0.02,
            xmax=plot_df["mean_bootstrap"],
            color="lightgray",
            alpha=0.8,
            linewidth=2
        )

    
        ax.errorbar(
            x=plot_df["mean_bootstrap"],
            y=y_pos,
            xerr=xerr,
            fmt="o",
            capsize=4,
            linewidth=1.5,
            markersize=6
        )


        for i, row in plot_df.iterrows():
            ax.text(
                row["ci_upper_95"] + 0.004,
                i,
                f'{row["mean_bootstrap"]:.3f} '
                f'[{row["ci_lower_95"]:.3f}; {row["ci_upper_95"]:.3f}]',
                va="center",
                fontsize=12,
                fontweight="bold" if row["Model"] == "LightGBM" else "normal"
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        ax.set_xlim(
            plot_df["ci_lower_95"].min() - 0.02,
            plot_df["ci_upper_95"].max() + 0.12
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(model_order, fontsize=12)
    axes[0].set_ylabel("Modellpipeline")


    plt.tight_layout()






def overview():

    df=pd.read_sql("SELECT * FROM model_results", con=engine)



    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass


    df=df[df["timestamp"].str.contains("2026-06-03")]
    df["rank"] = df.groupby(df.columns[0])["accuracy"].rank(ascending=False, method="first")
    df=df.sort_values(by=["accuracy"], ascending=False)
    df=df[df!="None"].fillna(0)
    df=df[df["rank"]==1.0]
    df.head()


    df=df.drop(columns=["rank", "timestamp","rank_overall"])


    print(df.columns)

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

    df.reset_index(drop=True)

    df = df.reset_index(drop=True)

    from great_tables import GT, loc, style

    df = df.reset_index(drop=True)

    gt_tbl = (
        GT(df)




        .tab_style(
            style=[
                style.text(size="30px", weight="bold")
            ],
            locations=loc.title())
    

        .tab_style(
            style=[
                style.text(size="25px"),
                style.text(weight="bold")
            ],
            locations=loc.column_labels()
        )
            .tab_style(
            style=[
                style.text(size="25px"),

            ],
            locations=loc.body()
        )

   
        .cols_width(
            **{col: "150px" for col in df.columns if col != "Training Time (s)" and col != "Model"},
        )

  
        .cols_align(
            align="center",
            columns=[col for col in df.columns if col != "Model"]
        )
        .cols_align(
            align="left",
            columns=["Model"]
        )
    )

    return gt_tbl


# %%

