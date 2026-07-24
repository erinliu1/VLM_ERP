from __future__ import annotations

import json
import os
import re
import time

import random
import numpy as np
import torch

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from are_you_using_cuda import is_using_cuda
is_using_cuda()

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from sentences import sentences

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PICTURES_DIR = Path("pictures")
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0},attn_implementation="sdpa")
model.eval()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
period_ids = processor.tokenizer.encode(".", add_special_tokens=False,)
if len(period_ids) != 1:
    raise ValueError(f"erin, um, i expected a single token ID for the period, but this model gave me {period_ids}")
PERIOD_TOKEN_ID = period_ids[0]


stimulus_set_hidden_states = {}

for item_index, item in enumerate(sentences):
    stimulus_set_hidden_states[item_index] = {
        'congruent': [],
        'incongruent': [],
    }

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
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        token_ids = inputs["input_ids"][0]
        period_index = (token_ids == PERIOD_TOKEN_ID).nonzero(as_tuple=True)[0][-1].item()
        critical_word_final_token_index = period_index - 1

        decoded_prefix = processor.tokenizer.decode(token_ids[:period_index].tolist(), skip_special_tokens=False)
        if not decoded_prefix.rstrip().endswith(final_word):
            raise ValueError(f"The token before the final period does not appear to belong to the critical word {final_word!r}.\nDecoded input before period: {decoded_prefix[-200:]!r}")

        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[1:] # exclude embedding state 0

        critical_word_hidden_states = torch.stack([layer_hidden_state[0, critical_word_final_token_index] for layer_hidden_state in hidden_states], dim=0).to(device="cpu", dtype=torch.float32)  # num_layers x hidden_size = 36 x 4096

        save_dict = {
            'image_word': image_word,
            'final_word': final_word,
            'hidden_states': critical_word_hidden_states,
        }

        if is_congruent:
            stimulus_set_hidden_states[item_index]['congruent'].append(save_dict)
        else:
            stimulus_set_hidden_states[item_index]['incongruent'].append(save_dict)

torch.save(stimulus_set_hidden_states, "pt_all.pt")

for item_index in stimulus_set_hidden_states.keys():
    for condition in ['congruent', 'incongruent']:
        for stimulus_item in stimulus_set_hidden_states[item_index][condition]:
            image_word = stimulus_item['image_word']
            hidden_states = stimulus_item['hidden_states']
            name = f"{item_index}_{image_word}_{condition}.pt"
            path = f"all_hidden_states/{name}"
            torch.save(hidden_states, path)