import pandas as pd

accuracy_df = pd.read_csv("cross_validation_data/permutation_layerwise_accuracies.csv")

max_stats_df = accuracy_df.groupby("seed", as_index=False)["all_accuracy"].max().rename(columns={"all_accuracy":"max_accuracy"})

max_stats_df.to_csv("cross_validation_data/max_stats.csv", index=False)