import json
import numpy as np
import pandas as pd 
from glob import glob

congruent_items, incongruent_items = [], []

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
        average_compatibility = np.mean(values)
        item_results = {
            'item_index': item_index,
            'sentence': sentence,
            'image': image_word,
            'average_compatibility': average_compatibility
        }
        if congruency == 'congruent':
            if average_compatibility > 3:
                item_results['pass'] = '✅'
            else:
                item_results['pass'] = '⚠️'
                item_results['scores'] = values
            congruent_items.append(item_results)
        elif congruency == 'incongruent':
            if average_compatibility <= 3:
                item_results['pass'] = '✅'
            else:
                item_results['pass'] = '⚠️'
                item_results['scores'] = values
            incongruent_items.append(item_results)
        
survey_results = {
    'congruent': congruent_items,
    'incongruent': incongruent_items
}

with open('qualtrics_results.json', 'w') as f:
    json.dump(survey_results, f, indent=2, ensure_ascii=False)
