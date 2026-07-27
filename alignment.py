import pandas as pd
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

vlm_ratings_path = Path(f"behavioral_verifications/behavior_{MODEL_ID.split('/')[-1].lower().replace('-', '_')}.csv")

vlm_ratings_df = pd.read_csv(vlm_ratings_path)

classifier_results_path = Path("cross_validation_data/results.csv")

classifier_results_df = pd.read_csv(classifier_results_path)

combined_df = classifier_results_df.merge(
    vlm_ratings_df,
    on=["item_index", "image_word", "condition"],
    how="left"
).drop(columns=["pass"])

combined_df = combined_df.rename(
    columns={"expected_rating": "vlm"}
)

coefficients = []

for layer_index in combined_df["layer_index"].unique():

    layer_df = combined_df[
        combined_df.layer_index == layer_index
    ].copy()

    layer_df["probability_z"] = (
        layer_df["probability_congruent"]
        - layer_df["probability_congruent"].mean()
    ) / layer_df["probability_congruent"].std()

    model = smf.ols(
        "vlm ~ probability_z + C(item_index)",
        data=layer_df
    ).fit()

    coefficients.append({
        "layer_index": layer_index,
        "coefficient": model.params["probability_z"],
        "se": model.bse["probability_z"],
        "p_value": model.pvalues["probability_z"]
    })

coefficients_df = pd.DataFrame(coefficients)
coefficients_df["p_value_fdr"] = multipletests(
    coefficients_df["p_value"],
    method="fdr_bh"
)[1]

coefficients_df["significant_fdr"] = (
    coefficients_df["p_value_fdr"] < 0.05
)
coefficients_df.to_csv(f"representation_behavioral_alignment/{MODEL_ID.split('/')[-1].lower().replace('-', '_')}.csv", index=False)