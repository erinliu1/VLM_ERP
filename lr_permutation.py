import os
import numpy as np
import pandas as pd
import torch
import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

with open("cv_lookup_pt.json", "r") as f:
    cv_lookup = json.load(f)

N_FOLDS = len(cv_lookup)

ALL_HIDDEN_STATES = {}
for filename in os.listdir("all_hidden_states"):
    ALL_HIDDEN_STATES[filename] = torch.load(f"all_hidden_states/{filename}", map_location="cpu", weights_only=False).float().numpy()

def unpack_item(item):
    parts = item.split("_")
    item_index = int(parts[0])
    image_word = parts[1]
    condition = parts[2].split(".")[0]
    return item_index, image_word, condition

def get_label(condition, reverse=False):
    if reverse:
        if condition == 'congruent':
            return 0
        else:
            return 1
    else:
        if condition == 'congruent':
            return 1
        else:
            return 0
    
def get_item(pt_list, reverse=False):
    X, y = [], []
    for item_name in pt_list:
        item_index, image_word, condition = unpack_item(item_name)
        hidden_state = ALL_HIDDEN_STATES[item_name]
        X.append(hidden_state)
        y.append(get_label(condition, reverse=reverse))
    return X, y

def run_permutation(SEED):
    OUTPUT_PATH = f"cross_validation_data/permutations/{SEED}.csv"
    if os.path.exists(OUTPUT_PATH):
        print(f'skipping {SEED}')
        return

    rng = np.random.default_rng(SEED)
    item_indices = np.arange(80)
    reverse_indices = set(item_indices[rng.random(len(item_indices)) < 0.5])

    results = []

    for fold in range(1, N_FOLDS + 1):
        train_items = cv_lookup[str(fold)]["train"]
        test_items = cv_lookup[str(fold)]["test"]

        X_train, y_train = [], []
        for item_index, pt_list in train_items.items():
            reverse = int(item_index) in reverse_indices
            X_item, y_item = get_item(pt_list, reverse=reverse)
            X_train.extend(X_item)
            y_train.extend(y_item)
                
        X = np.stack(X_train, axis=0) # (288, 36, 4096)
        y = np.asarray(y_train, dtype=np.int64) # (288,)

        classifiers = {}

        N_LAYERS = X.shape[1]
        
        for layer_index in range(N_LAYERS):
            X_layer = X[:, layer_index, :] # (288, 4096)

            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    solver="liblinear",
                    max_iter=10_000,
                    random_state=SEED,
                ),
            )

            classifier.fit(X_layer, y)
            classifiers[layer_index] = classifier

        for item_index, pt_list in test_items.items():
            reverse = int(item_index) in reverse_indices
            for item_name in pt_list:
                _, image_word, condition = unpack_item(item_name)
                X_test = ALL_HIDDEN_STATES[item_name] # (36, 4096)
                label = get_label(condition, reverse=reverse)
                condition = 'congruent' if label == 1 else 'incongruent'
                
                for layer_index, classifier in classifiers.items():
                    X_test_layer = X_test[layer_index, :].reshape(1, -1) # (1, 4096)
                    predicted_label = int(classifier.predict(X_test_layer)[0])
                    probability_congruent = float(classifier.predict_proba(X_test_layer)[0][1])
                    results.append({
                        'layer_index': layer_index,
                        'item_index': item_index,
                        'image_word': image_word,
                        'condition': condition,
                        'pass': '✅' if predicted_label == label else '❌',
                        'probability_congruent': probability_congruent
                    })

    df = pd.DataFrame(results)
    df = df.sort_values(by=['layer_index', 'item_index']).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

from concurrent.futures import ProcessPoolExecutor

N_PERMUTATIONS = 1000

if __name__ == "__main__":
    permutations = range(N_PERMUTATIONS)
    with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        list(executor.map(run_permutation, permutations))