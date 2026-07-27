import pandas as pd
import numpy as np

observed_df = pd.read_csv("cross_validation_data/layerwise_accuracy.csv")
max_stats_df = pd.read_csv("cross_validation_data/max_stats.csv")

null = max_stats_df["max_accuracy"].to_numpy()
threshold_95 = np.quantile(null, 0.95)

print(f"95th percentile of null distribution: {threshold_95}")

p_values, significant = [], []
for _, row in observed_df.iterrows():
    obs = row["all_accuracy"]
    p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    p_values.append(p)
    significant.append(obs > threshold_95)

observed_df["maxstat_p"] = p_values
observed_df["exceeds_95th_percentile"] = significant

print(
    observed_df[
        ["layer_index", "all_accuracy", "maxstat_p", "exceeds_95th_percentile"]
    ]
)

