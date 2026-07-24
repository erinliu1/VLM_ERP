import os
import json

with open("cv_indices.json", "r") as f:
    cross_validation_split = json.load(f)

def unpack_item(item):
    parts = item.split("_")
    item_index = int(parts[0])
    image_word = parts[1]
    condition = parts[2].split(".")[0]
    return item_index, image_word, condition

item_to_pt = {i: [] for i in range(80)}

for filename in os.listdir('all_hidden_states'):
    item_index, image_word, condition = unpack_item(filename)
    item_to_pt[item_index].append(filename)    

# save the dictionary to a json file
with open('pt_item_lookup.json', 'w') as f:
    json.dump(item_to_pt, f)