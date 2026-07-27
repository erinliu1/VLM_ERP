import json
import numpy as np
import pandas as pd 
from glob import glob

OUTPUT_PATH = f"behavioral_verifications/behavior_human.csv"

# average_ratings = []
# for item_index, item in enumerate(sentences):
#     word_a, word_b = item['word_options']
    
#     congruent_pairs = [(word_a, word_a), (word_b, word_b)]
#     incongruent_pairs = [(word_a, word_b), (word_b, word_a)]

#     for image_word, final_word in congruent_pairs + incongruent_pairs:
#         is_congruent = (image_word == final_word)

#         average_rating = # ##
#         average_ratings.append({
#             "item_index": item_index,
#             "image_word": image_word,
#             "condition": "congruent" if is_congruent else "incongruent",
#             "average_rating": average_rating,
#         })


all_survey_results = []
for survey_ID in ['A', 'B', 'C', 'D']:
    survey_json_path = f"survey_assignments/survey_{survey_ID}.json"
    survey_response_path = glob(f"survey_responses/Survey {survey_ID}_*.csv")[0]

    with open(survey_json_path, 'r') as f:
        survey_data = json.load(f)

    survey_questions = survey_data['questions']
    survey_response = pd.read_csv(survey_response_path)

    for question in survey_questions:
        question_id = question['question_number']
        congruency = question['congruency']
        item_index = question['item_index']
        image_word = question['image_filename'].split('.')[0]
        sentence = question['sentence']
        vals = survey_response[f'S{survey_ID}_Q{question_id}']
        values = [int(v[0]) for v in vals if v[0] in ['1','2','3','4','5']]
        average_rating = np.mean(values)
        all_survey_results.append({
            'item_index': item_index,
            'image_word': image_word,
            'condition': congruency,
            'average_rating': average_rating
        })

df = pd.DataFrame(all_survey_results)
df = df.sort_values(by=['item_index', 'condition', 'image_word']).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)
