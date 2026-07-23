import os
import numpy as np
import torch
import json
from sklearn.model_selection import KFold

OUPUT_PATH = "cross_validation_data"
os.makedirs(OUPUT_PATH, exist_ok=True)

stimulus_set_hidden_states = torch.load(
    "stimulus_set_hidden_states.pt",
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
    fold_dir = os.path.join(OUPUT_PATH, f"fold_{fold_index}")
    os.makedirs(fold_dir, exist_ok=True)
    cross_validation_split[fold_index] = {
        "train": train_indices.tolist(),
        "test": test_indices.tolist(),
    }
    fold_test_dir = os.path.join(fold_dir, "test")
    os.makedirs(fold_test_dir, exist_ok=True)

    for test_index in test_indices:
        for condition in ['congruent', 'incongruent']:
            for stimulus_item in stimulus_set_hidden_states[test_index][condition]:
                image_word = stimulus_item['image_word']
                hidden_state = stimulus_item['hidden_states']
                test_item_path = os.path.join(fold_test_dir, f"{test_index}_{image_word}_{condition}.pt")
                torch.save(hidden_state, test_item_path)

    fold_train_dir = os.path.join(fold_dir, "train")
    os.makedirs(fold_train_dir, exist_ok=True)

    train_congruent, train_incongruent = [], []
    for train_index in train_indices:
        for stimulus_item in stimulus_set_hidden_states[train_index]['congruent']:
            train_congruent.append(stimulus_item['hidden_states'])
        for stimulus_item in stimulus_set_hidden_states[train_index]['incongruent']:
            train_incongruent.append(stimulus_item['hidden_states'])
    train_congruent = torch.stack(train_congruent, dim=0)
    train_incongruent = torch.stack(train_incongruent, dim=0)
    torch.save(train_congruent, os.path.join(fold_train_dir, "train_congruent.pt"))
    torch.save(train_incongruent, os.path.join(fold_train_dir, "train_incongruent.pt"))

with open(os.path.join(OUPUT_PATH, "cross_validation_split.json"), "w", encoding="utf-8") as f:
    json.dump(cross_validation_split, f, indent=4)

