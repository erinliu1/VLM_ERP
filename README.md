stimulus_set_hidden_states.pt is a dictionary

main keys are item indices (0-79)

within each item, keys are 'congruent' and 'incongruent' which are lists of two elements each, both of which are dictionaries with keys image_word, final_word, hidden_states

in the all_hidden_states folder, you'll find the (36 by 4096) hidden states vector for each sample in each stimulus set

cross_validation_split.json