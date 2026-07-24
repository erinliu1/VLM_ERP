import os
import numpy as np
import torch
import json
from sklearn.model_selection import KFold

stimulus_set_hidden_states = torch.load(
    "pt_all.pt",
    map_location="cpu",
    weights_only=False,
)

SEED = 42
N_SPLITS = 10

cross_validation_split = {}

item_indices = list(stimulus_set_hidden_states.keys())
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold_index, (train_indices, test_indices) in enumerate(kf.split(item_indices)):
    fold_index = fold_index + 1
    cross_validation_split[fold_index] = {
        "train": train_indices.tolist(),
        "test": test_indices.tolist(),
    }

with open("cv_indices.json", "w", encoding="utf-8") as f:
    json.dump(cross_validation_split, f, indent=4)

