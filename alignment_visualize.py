import pandas as pd
import matplotlib.pyplot as plt
import os

dependent_var = 'Human' # or VLM
alignment_df = pd.read_csv(
    "representation_behavioral_alignment/qwen3_vl_8b_instruct.csv"
)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_behavioral_alignment():

    plot_df = alignment_df.copy()

    plot_df["lower"] = (
        plot_df["coefficient"] - 1.96 * plot_df["se"]
    )

    plot_df["upper"] = (
        plot_df["coefficient"] + 1.96 * plot_df["se"]
    )

    plt.figure(figsize=(10, 5))


    # zero line
    plt.axhline(
        0,
        linestyle="--",
        color="gray",
        label="no alignment"
    )


    # stable alignment region
    plt.axvspan(
        13,
        35,
        color="gray",
        alpha=0.15,
        label="stable regime"
    )


    # coefficient
    plt.plot(
        plot_df["layer_index"],
        plot_df["coefficient"],
        marker="o",
        color="tab:blue",
        label="coefficient"
    )


    # CI
    plt.fill_between(
        plot_df["layer_index"],
        plot_df["lower"],
        plot_df["upper"],
        color="tab:blue",
        alpha=0.2,
        label="95% CI"
    )


    plt.xlabel("Layer")
    plt.ylabel(
        "Regression Coefficient"
    )

    plt.title(
        "Alignment Between VLM Representations and VLM Behavioral Ratings"
    )


    plt.legend(
        loc="lower right",
        framealpha=0.9,
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"{PLOTS_DIR}/behavioral_alignment_with_stats.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


plot_behavioral_alignment()