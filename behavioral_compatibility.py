import json
from collections import Counter

with open("qualtrics_results.json", "r") as f:
    data = json.load(f)

congruent_items = data["congruent"]

counts = Counter(item["pass"] for item in congruent_items)

print(f"Actually Congruent: {counts}")
print()

incongruent_items = data["incongruent"]

counts = Counter(item["pass"] for item in incongruent_items)

print(f"Actually Incongruent: {counts}")
