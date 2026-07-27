from dotenv import load_dotenv
load_dotenv()

from are_you_using_cuda import is_using_cuda
is_using_cuda()

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from sentences import sentences


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PICTURES_DIR = Path("pictures")

OUTPUT_PATH = Path(f"behavioral_verifications/behavior_{MODEL_ID.split('/')[-1].lower().replace('-', '_')}.csv")

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0},attn_implementation="sdpa")
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

SYSTEM_PROMPT = """
You will be shown one picture and one sentence. Your task is to judge how compatible the final word of the sentence is with the situation shown in the picture.

When making your judgment:
* Focus on the final word of the sentence. Treat the rest of the sentence as given.
* The picture does not need to prove that the sentence ending is the only possible interpretation. Simply, decide whether the final word is a natural and reasonable interpretation of the picture.

Use the following rating scale:
1 - Not compatible at all. The final word is clearly inconsistent with / unrelated to / contradicts the picture.
2 - Low compatibility. The final word is technically plausible but would not be commonly associated with the picture / it would be unusual or difficult to reconcile this sentence with the picture.
3 - Uncertain. The final word provides weak / ambiguous / mixed evidence for the picture.
4 - Compatible. The final word makes the sentence a reasonable interpretation of the picture.
5 - Very compatible. The final word makes the sentence very consistent with / a natural interpretation of the picture.

Respond with exactly one token: 1, 2, 3, 4, or 5. Do not output any additional text.
""".strip()

RATING_TOKEN_IDS = [processor.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

expected_ratings = []
for item_index, item in enumerate(tqdm(sentences, desc="Rating")):
    sentence_frame = item['sentence_frame']
    word_a, word_b = item['word_options']
    
    congruent_pairs = [(word_a, word_a), (word_b, word_b)]
    incongruent_pairs = [(word_a, word_b), (word_b, word_a)]

    for image_word, final_word in congruent_pairs + incongruent_pairs:
        is_congruent = (image_word == final_word)

        sentence = f"{sentence_frame} {final_word}."
        image_path = PICTURES_DIR / f"{image_word}.png"
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": sentence,
                    },
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1]
        rating_logits = next_token_logits[RATING_TOKEN_IDS]
        probabilities = F.softmax(rating_logits, dim=0)
        ratings = torch.arange(1, 6, device=probabilities.device, dtype=probabilities.dtype)
        expected_rating = (probabilities * ratings).sum().item()
        
        expected_ratings.append({
            "item_index": item_index,
            "image_word": image_word,
            "condition": "congruent" if is_congruent else "incongruent",
            "expected_rating": expected_rating,
        })

df = pd.DataFrame(expected_ratings)
df = df.sort_values(by=['item_index', 'condition', 'image_word']).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)