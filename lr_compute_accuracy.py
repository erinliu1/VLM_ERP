import numpy as np
import pandas as pd


INPUT_PATH = "cross_validation_data/results.csv"
OUTPUT_PATH = "cross_validation_data/layerwise_accuracy.csv"

results_df = pd.read_csv(INPUT_PATH)

layerwise_accuracy = []
layers = results_df["layer_index"].unique()
for layer_index in layers:
    layer_df = results_df[results_df["layer_index"] == layer_index]
    n_correct = (layer_df["pass"] == "✅").sum()
    n_total = len(layer_df)
    accuracy = n_correct / n_total
    congruent_df = layer_df[layer_df["condition"] == "congruent"]
    n_congruent_correct = (congruent_df["pass"] == "✅").sum()
    n_congruent_total = len(congruent_df)
    congruent_accuracy = n_congruent_correct / n_congruent_total
    incongruent_df = layer_df[layer_df["condition"] == "incongruent"]
    n_incongruent_correct = (incongruent_df["pass"] == "✅").sum()
    n_incongruent_total = len(incongruent_df)
    incongruent_accuracy = n_incongruent_correct / n_incongruent_total
    layerwise_accuracy.append({
        "layer_index": layer_index,
        "all_accuracy": accuracy,
        "congruent_accuracy": congruent_accuracy,
        "incongruent_accuracy": incongruent_accuracy,
    })

accuracy_df = pd.DataFrame(layerwise_accuracy)
accuracy_df.to_csv(OUTPUT_PATH, index=False)