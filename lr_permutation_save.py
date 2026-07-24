import glob
import numpy as np
import pandas as pd

files = glob.glob("cross_validation_data/permutations/*.csv")
OUTPUT_PATH = "cross_validation_data/permutation_layerwise_accuracies.csv"

accuracies = []
for f in files:
    results_df = pd.read_csv(f)
    seed = int(f.split("/")[-1].split(".")[0])
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
        accuracies.append({
            "seed": seed,
            "layer_index": layer_index,
            "all_accuracy": accuracy,
            "congruent_accuracy": congruent_accuracy,
            "incongruent_accuracy": incongruent_accuracy,
        })

accuracy_df = pd.DataFrame(accuracies)
accuracy_df = accuracy_df.sort_values(by=["seed", "layer_index"]).reset_index(drop=True)
accuracy_df.to_csv(OUTPUT_PATH, index=False)

