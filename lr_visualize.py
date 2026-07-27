import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

main_accuracy_df = pd.read_csv(
    "cross_validation_data/layerwise_accuracy.csv"
)

bootstrap_accuracy_df = pd.read_csv(
    "cross_validation_data/all_bootstraps.csv"
)

max_stats_df = pd.read_csv(
    "cross_validation_data/max_stats.csv"
)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_accuracy_with_stats(metric, title, color):

    # -------------------------
    # Bootstrap CI
    # -------------------------
    bootstrap_ci = (
        bootstrap_accuracy_df
        .groupby("layer_index")[metric]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
    )

    bootstrap_ci.columns = [
        "layer_index",
        "lower",
        "upper"
    ]

    plot_df = main_accuracy_df.merge(
        bootstrap_ci,
        on="layer_index"
    )


    # -------------------------
    # Permutation threshold
    # -------------------------
    # 95th percentile of maximum null accuracy
    permutation_threshold = (
        max_stats_df["max_accuracy"]
        .quantile(0.95)
    )


    # -------------------------
    # Peak band
    # -------------------------
    peak_threshold = 0.95

    peak_layers = plot_df.loc[
        plot_df[metric] >= peak_threshold,
        "layer_index"
    ]

    peak_start = peak_layers.min()
    peak_end = peak_layers.max()

    peak_layers_exist = len(peak_layers) > 0


    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(10, 5))


    # Permutation FWER threshold
    plt.axhline(
        permutation_threshold,
        linestyle="--",
        color="gray",
        label=f"significance threshold ({permutation_threshold:.3f})"
    )


    # Highlight peak band
    if peak_layers_exist:
        plt.axvspan(
            peak_start,
            peak_end,
            color="gray",
            alpha=0.15,
            label=f"accuracy ≥ {peak_threshold}"
        )


    # Accuracy
    plt.plot(
        plot_df["layer_index"],
        plot_df[metric],
        marker="o",
        color=color,
        label="Observed accuracy"
    )


    # Bootstrap CI
    plt.fill_between(
        plot_df["layer_index"],
        plot_df["lower"],
        plot_df["upper"],
        color=color,
        alpha=0.2,
        label="95% bootstrap CI"
    )


    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.ylim(0.45, 1.0)

    plt.legend(
        loc="lower right",
        framealpha=0.9,
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{PLOTS_DIR}/{metric}_with_stats.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


plot_accuracy_with_stats(
    "all_accuracy",
    "Congruence Decoding Accuracy by Layer",
    "tab:blue"
)