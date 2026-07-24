from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

INPUT_PATH = "cross_validation_data/results.csv"
OUTPUT_DIR = "cross_validation_data/bootstraps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
item_groups = {item: group for item, group in df.groupby("item_index")}

def run_bootstrap(SEED):
    OUTPUT_PATH = f"{OUTPUT_DIR}/{SEED}.csv"
    if os.path.exists(OUTPUT_PATH):
        return

    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.choice(np.arange(80), size=80, replace=True)

    bootstrap_df = pd.concat([item_groups[i] for i in bootstrap_indices], ignore_index=True)
    bootstrap_df = bootstrap_df.sort_values(by=["layer_index", "item_index"]).reset_index(drop=True)

    layerwise_accuracy = []
    layers = bootstrap_df["layer_index"].unique()
    for layer_index in layers:
        layer_df = bootstrap_df[bootstrap_df["layer_index"] == layer_index]
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

    bootstrap_df = pd.DataFrame(layerwise_accuracy)
    bootstrap_df.to_csv(OUTPUT_PATH, index=False)

from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    N_BOOTSTRAPS = 10000
    N_WORKERS = 64

    seeds = range(N_BOOTSTRAPS)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        list(executor.map(run_bootstrap, seeds))