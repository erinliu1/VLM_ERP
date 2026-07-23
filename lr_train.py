import os
import numpy as np
import torch
import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 10
OUTPUT_PATH = "cross_validation_data/results.json"

predictions = None

def unpack_test_item(test_item):
    parts = test_item.split("_")
    item_index = int(parts[0])
    image_word = parts[1]
    condition = parts[2].split(".")[0]
    return item_index, image_word, condition

def is_correct(condition, prediction):
    if condition == 'congruent':
        return prediction == 1
    elif condition == 'incongruent':
        return prediction == 0
    else:
        raise ValueError(f"Unknown condition: {condition}")

for fold in range(1, N_FOLDS + 1):
    print(f"Processing fold {fold}...")
    INPUT_FOLDER = f"cross_validation_data/fold_{fold}"

    X_congruent = torch.load(f"{INPUT_FOLDER}/train/train_congruent.pt", map_location="cpu", weights_only=False)
    X_incongruent = torch.load(f"{INPUT_FOLDER}/train/train_incongruent.pt", map_location="cpu", weights_only=False)

    N_LAYERS = X_congruent.shape[1]

    if predictions is None:
        predictions = [{
            'layer_index': layer_index,
            'congruent': [],
            'incongruent': [],
        } for layer_index in range(N_LAYERS)]
    
    X = torch.cat([X_congruent, X_incongruent], dim=0)
    y = torch.cat([
        torch.ones(X_congruent.shape[0], dtype=torch.long),     # congruent label = 1
        torch.zeros(X_incongruent.shape[0], dtype=torch.long),  # incongruent label = 0
    ])

    X = X.float().numpy() # (288, 36, 4096)
    y = y.numpy() # (288,)

    classifiers = {}

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

    fold_test_folder = f"{INPUT_FOLDER}/test"

    for test_item in sorted(os.listdir(fold_test_folder)):
        item_index, image_word, condition = unpack_test_item(test_item)
        X_test = torch.load(f"{fold_test_folder}/{test_item}", map_location="cpu", weights_only=False).float().numpy() # (36, 4096)

        for layer_index, classifier in classifiers.items():
            X_test_layer = X_test[layer_index, :].reshape(1, -1) # (1, 4096)

            predicted_label = int(classifier.predict(X_test_layer)[0])
            probability_congruent = float(classifier.predict_proba(X_test_layer)[0][1])
            
            predictions[layer_index][condition].append({
                'item_index': item_index,
                'image_word': image_word,
                'pass': '✅' if is_correct(condition, predicted_label) else '❌',
                'probability_congruent': probability_congruent
            })

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(predictions, file, indent=2, ensure_ascii=False)

