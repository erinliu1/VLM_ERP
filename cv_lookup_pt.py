import json

with open("cv_indices.json", "r") as f:
    cv_indices = json.load(f)

with open("pt_item_lookup.json", "r") as f:
    pt_item_lookup = json.load(f)

cv_lookup_pt = {}
for fold_index, indices in cv_indices.items():
    fold_index = int(fold_index)
    cv_lookup_pt[fold_index] = {
        "train": {},
        "test": {},
    }
    train_indices = indices["train"]
    test_indices = indices["test"]
    for item_index in train_indices:
        cv_lookup_pt[fold_index]["train"][item_index] = pt_item_lookup[str(item_index)]
    for item_index in test_indices:
        cv_lookup_pt[fold_index]["test"][item_index] = pt_item_lookup[str(item_index)]

# print(cv_lookup_pt.keys()) -> dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(cv_lookup_pt[1].keys()) -> dict_keys(['train', 'test'])
# print(cv_lookup_pt[1]["train"].keys()) -> dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79])
# print(cv_lookup_pt[1]["train"][1]) -> ['1_disappointment_congruent.pt', '1_delight_congruent.pt', '1_disappointment_incongruent.pt', '1_delight_incongruent.pt']

with open("cv_lookup_pt.json", "w") as f:
    json.dump(cv_lookup_pt, f)