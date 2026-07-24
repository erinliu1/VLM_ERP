import pandas as pd
import numpy as np

observed_df = pd.read_csv("cross_validation_data/layerwise_accuracy.csv")
max_stats_df = pd.read_csv("cross_validation_data/max_stats.csv")

null = max_stats_df["max_accuracy"].to_numpy()

p_values = []
for _, row in observed_df.iterrows():
    obs = row["all_accuracy"]
    p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    p_values.append(p)

observed_df["maxstat_p"] = p_values

print(observed_df)