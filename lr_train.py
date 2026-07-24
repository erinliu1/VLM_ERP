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

results = []
N_FOLDS = len(cv_lookup)
SEED = 42
OUTPUT_PATH = f"cross_validation_data/results.csv"
reverse_indices = [] # define per permutation

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
        hidden_state = torch.load(f"all_hidden_states/{item_name}", map_location="cpu", weights_only=False)
        X.append(hidden_state)
        y.append(get_label(condition, reverse=reverse))
    return X, y

for fold in range(1, N_FOLDS + 1):
    train_items = cv_lookup[str(fold)]["train"]
    test_items = cv_lookup[str(fold)]["test"]

    X_train, y_train = [], []
    for item_index, pt_list in train_items.items():
        reverse = item_index in reverse_indices
        X_item, y_item = get_item(pt_list, reverse=reverse)
        X_train.extend(X_item)
        y_train.extend(y_item)
    
    X_train = torch.stack(X_train, dim=0)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X = X_train.float().numpy() # (288, 36, 4096)
    y = y_train.numpy() # (288,)

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
        reverse = item_index in reverse_indices
        for item_name in pt_list:
            _, image_word, condition = unpack_item(item_name)
            X_test = torch.load(f"all_hidden_states/{item_name}", map_location="cpu", weights_only=False).float().numpy() # (36, 4096)
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




    
        