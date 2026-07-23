from __future__ import annotations

import json
import numpy as np
import pandas as pd

with open("cross_validation_data/layerwise_accuracy.json", "r") as f:
    results = json.load(f)

SEED = 1

rng = np.random.default_rng(SEED)

bootstrap_indices = rng.choice(np.arange(80), size=80, replace=True)

layerwise_accuracy = []
for layer_data in results:
    df = pd.DataFrame(layer_data['congruent'] + layer_data['incongruent'])
    print(df.head())
    break        
