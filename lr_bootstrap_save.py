import glob
import pandas as pd

files = glob.glob("cross_validation_data/bootstraps/*.csv")

all_bootstrap_dfs = []
for f in files:
    df = pd.read_csv(f)
    seed = int(f.split("/")[-1].split(".")[0])
    df["seed"] = seed
    all_bootstrap_dfs.append(df)

all_bootstraps = pd.concat(all_bootstrap_dfs, ignore_index=True)
all_bootstraps = all_bootstraps.sort_values(by=["seed", "layer_index"]).reset_index(drop=True)

all_bootstraps.to_csv("cross_validation_data/all_bootstraps.csv", index=False)