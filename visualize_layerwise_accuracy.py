import json
import matplotlib.pyplot as plt

with open("cross_validation_data/layerwise_accuracy.json", "r") as f:
    results = json.load(f)

layers = [r["layer_index"] + 1 for r in results]
overall = [r["all_accuracy"] for r in results]
congruent = [r["congruent_accuracy"] for r in results]
incongruent = [r["incongruent_accuracy"] for r in results]

plt.figure(figsize=(8,4.5))

plt.plot(layers, overall, lw=2.5, label="Overall")
plt.plot(layers, congruent, lw=2, label="Congruent")
plt.plot(layers, incongruent, lw=2, label="Incongruent")

plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)

plt.xlabel("Layer")
plt.ylabel("Classification Accuracy")
plt.xticks(range(1, 37, 5))
plt.xlim(1, 36)
plt.ylim(0.45, 1.0)

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/layerwise_accuracy.png", dpi=300)