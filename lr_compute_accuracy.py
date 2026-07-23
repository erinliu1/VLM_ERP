import numpy as np
import json
from collections import Counter


PREDICTIONS_PATH = "cross_validation_data/results.json"
OUTPUT_PATH = "cross_validation_data/layerwise_accuracy.json"


with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
    predictions = json.load(file)

layerwise_accuracy = []

def compute_accuracy(counts):
    return counts['✅'] / (counts['✅'] + counts['❌'])

for layer_data in predictions:
    layer_index = layer_data['layer_index']
    
    all_predictions = layer_data['congruent'] + layer_data['incongruent']
    all_counts = Counter(item["pass"] for item in all_predictions)

    congruency_counts = Counter(item["pass"] for item in layer_data['congruent'])

    incongruency_counts = Counter(item["pass"] for item in layer_data['incongruent'])

    layerwise_accuracy.append({
        'layer_index': layer_index,
        'all_accuracy': compute_accuracy(all_counts),
        'congruent_accuracy': compute_accuracy(congruency_counts),
        'incongruent_accuracy': compute_accuracy(incongruency_counts),
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(layerwise_accuracy, file, indent=2, ensure_ascii=False)
