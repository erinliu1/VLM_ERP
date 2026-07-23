from __future__ import annotations

import json
import os
import re
import time

import numpy as np
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from are_you_using_cuda import is_using_cuda
is_using_cuda()

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from sentences import sentences


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PICTURES_DIR = Path("pictures")
OUTPUT_PATH = Path(f"behavior_{MODEL_ID.split('/')[-1].lower().replace('-', '_')}.json")
MAX_TOKENS = 100
MAX_SENTENCES = None
MAX_ATTEMPTS = 4
REQUEST_DELAY_SECONDS = 0.5

SAMPLES_PER_PROMPT = 20

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0},attn_implementation="sdpa")
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

SYSTEM_PROMPT = """
You are evaluating stimuli for a picture-sentence semantic compatibility experiment.

You will receive:

1. One picture.
2. One sentence ending in a critical final word.

Your taks is to judge how compatible the sentence's final word is with the situation shown in the picture.

IMPORTANT RULES

1. Evaluate the compatibility of the sentence ending in the context of the entire sentence. The words before the final word provide the context for interpreting the critical word and should be assumed true, i.e. do not judge whether earlier parts of the sentence are true by themselves. For example, for a sentence like "A woman was upset while driving," do not rely on information about whether the person in the picture is a woman or whether she looks upset to determine compatibility. You must evaluate the final word, i.e. driving, to see if that is consistent with the picture. 

That being said, final words cannot be interpreted in complete isolation without the sentence context. If the sentence establishes a relationship, location, object, destination, role, or other concept that directly impacts the interpretation of the final word, you should evaluate whether the picture supports the entire sentiment conveyed by the sentence involving that word.

Furthermore, for identities established in the picture or sentence, use common sense (boy, girl, man, woman, teacher, doctor, etc.) to infer the most likely interpretation of the identity. 

2. Examine objects, actions, events, settings, and relationships in the picture that are relevant to the sentence's final word. 

3. You may consider facial expressions, body language, and situational context from the picture, but all inferences must be grounded in visible evidence. Be very careful when inferring facial expression and body language to ensure you are making an accurate assessment.

4. Do not invent new objects, actions, events, locations, or relationships. Do not assume family relationships, occupations, identities, intentions, or events being common sense.

5. Do not require the sentence ending to be literally proven by the picture, and do not require the final word to be literally depicted in the picture. The picture only needs to provide enough information for the sentence ending to be reasonable; it does not need to uniquely determine that specific sentence ending. Use everyday semantic associations, categorical membership, and common sense to reason about compatibility. 

However, the picture should provide visual evidence that is semantically supported by the sentence ending. If the picture is largely irrelevant to the sentence, do not assign high compatibility just because the picture does not directly contradict the sentence. A rating of 4 or 5 should require that the picture suggests that situation is at least reasonable. Reserve low ratings (1 or 2) for final words that are genuinely inconsistent with, contradicted by, highly unrelated to, or irrelevant to the picture.

6. Do not give high ratings if you require convoluted reasoning to justify the final word. 

7. Use common sense and do not overthink. 

Use this rating scale:

   1 = Not compatible at all. The final word is clearly inconsistent with / unrelated to / contradicts the picture.

   2 = Low compatibility. The final word is technically plausible but would not be commonly associated with the picture / it would be unusual or difficult to reconcile this sentence with the picture. 

   3 = Uncertain. The final word provides weak / ambiguous / mixed evidence for the picture.

   4 = Compatible. The final word makes the sentence a reasonable interpretation of the picture.

   5 = Very compatible. The final word makes the sentence very consistent with / a natural interpretation of the picture.

Return exactly one JSON object with this structure:

{
  "compatibility": 1,
  "explanation": "Brief 1 sentence explanation grounded in visible evidence and its relation to the final word."
}

The compatibility value must be an integer from 1 through 5.
Do not include Markdown formatting or text outside the JSON object.
""".strip()

def strip_markdown_fences(content):
    """Remove common Markdown JSON fences if the model adds them."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
        
def extract_json_object(content):
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    if start == -1:
        raise ValueError(f"The model did not return a JSON object: {content}")

    candidate = content[start:].strip()

    # Repair a response that contains a complete object but omits
    # the final closing brace.
    if candidate.startswith("{") and not candidate.endswith("}"):
        candidate += "}"

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(f"The model returned malformed JSON: {content}") from error
        
def parse_model_response(content):
    cleaned = strip_markdown_fences(content)
    parsed = extract_json_object(cleaned)
    compatibility = parsed.get("compatibility")
    explanation = parsed.get("explanation")
    if not isinstance(compatibility, int) or not (1 <= compatibility <= 5):
        raise ValueError(f"Invalid compatibility value: {compatibility}")
    if not isinstance(explanation, str):
        raise ValueError(f"Invalid explanation value: {explanation}")
    return compatibility, explanation.strip()

def rate_compatibility(image_path, sentence_frame, word_option):
    image = Image.open(image_path).convert("RGB")
    sentence = f"{sentence_frame} {word_option}."
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
                    "text": f"Sentence: {sentence}\n\n Critical final word: {word_option}"

                },
            ],
        },
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=SAMPLES_PER_PROMPT,
        max_new_tokens=MAX_TOKENS,
    )    
    prompt_length = inputs["input_ids"].shape[1]
    response_ids = generated_ids[:, prompt_length:]
    responses = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    compatibilities, explanations = [], []
    for response_text in responses:
        compatibility, explanation = parse_model_response(response_text)
        compatibilities.append(compatibility)
        explanations.append(explanation)
    return compatibilities, explanations

def rate_compatibility_with_retries(image_path, sentence_frame, word_option):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return rate_compatibility(image_path, sentence_frame, word_option)
        except Exception as error:
            last_error = error
            if attempt == MAX_ATTEMPTS:
                break
            print(f"Attempt {attempt}/{MAX_ATTEMPTS} failed: {error}. Retrying in {REQUEST_DELAY_SECONDS} seconds...")
            time.sleep(REQUEST_DELAY_SECONDS)
    raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed. Last error: {last_error}")

def load_existing_results():
    if not OUTPUT_PATH.exists():
        return {"congruent": [], "incongruent": [], "errors": []}

    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            results = json.load(file)
    except Exception as _:
        return {"congruent": [], "incongruent": [], "errors": []}

    results.setdefault("congruent", [])
    results.setdefault("incongruent", [])
    results.setdefault("errors", [])

    results["congruent"].sort(key=lambda x: x["item_index"])
    results["incongruent"].sort(key=lambda x: x["item_index"])
    results["errors"].sort(key=lambda x: x["item_index"])

    save_output(results["congruent"], results["incongruent"], results["errors"],)

    return results

def get_completed(results):
    completed_congruent, completed_incongruent = set(), set()
    for item in results['congruent']:
        item_index = item.get("item_index")
        image_word = item.get("image")
        completed_congruent.add(f"{item_index}:{image_word}")
    for item in results['incongruent']:
        item_index = item.get("item_index")
        image_word = item.get("image")
        completed_incongruent.add(f"{item_index}:{image_word}")
    return completed_congruent, completed_incongruent

def save_output(congruent_items, incongruent_items, errors):
    congruent_items = sorted(congruent_items, key=lambda x: x["item_index"])
    incongruent_items = sorted(incongruent_items, key=lambda x: x["item_index"])
    errors = sorted(errors, key=lambda x: x["item_index"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump({"congruent": congruent_items, "incongruent": incongruent_items, "errors": errors}, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(OUTPUT_PATH)

selected_sentences = sentences if MAX_SENTENCES is None else sentences[:MAX_SENTENCES]
results = load_existing_results()
completed_congruent, completed_incongruent = get_completed(results)
    
congruent_items, incongruent_items, errors = results["congruent"], results["incongruent"], results["errors"]

def check_compatibility_rating(compatibilities, explanations, condition):
    average_compatibility = np.mean(compatibilities)
    issues = []
    if condition == 'congruent':
        if average_compatibility > 3:
            print(f"✅ Congruent: {average_compatibility}")  
        else:
            print(f"⚠️ Congruent but rated: {average_compatibility}")
            for i, (compatibility, explanation) in enumerate(zip(compatibilities, explanations)):
                if compatibility <= 3:
                    issues.append({'compatibility': compatibility, 'explanation': explanation})
                    print(f"    sample {i} rated {compatibility}: {explanation}")
    elif condition == 'incongruent':
        if average_compatibility <= 3:
            print(f"✅ Incongruent: {average_compatibility}")
        else:
            print(f"⚠️ Incongruent but rated: {average_compatibility}")
            for i, (compatibility, explanation) in enumerate(zip(compatibilities, explanations)):
                if compatibility > 3:
                    issues.append({'compatibility': compatibility, 'explanation': explanation})
                    print(f"    sample {i} rated {compatibility}: {explanation}")
    return average_compatibility, issues

for i, item in enumerate(selected_sentences):
    sentence_frame = item["sentence_frame"]
    word_a, word_b = item["word_options"]
    congruent_pairings = [(word_a, word_a), (word_b, word_b)]
    incongruent_pairings = [(word_a, word_b), (word_b, word_a)]
    for image_word, final_word in congruent_pairings:
        key = f"{i}:{image_word}"
        if key in completed_congruent:
            continue
        sentence = f"{sentence_frame} {final_word}."
        print(f"Sentence {i}: {sentence}")
        try:
            image_path = f"{PICTURES_DIR}/{image_word}.png"
            compatibilities, explanations = rate_compatibility_with_retries(image_path, sentence_frame, final_word)
            average_compatibility, issues = check_compatibility_rating(compatibilities, explanations, 'congruent')
            congruent_items.append({
                "item_index": i,
                "sentence": sentence,
                "image": image_word,
                "average_compatibility": average_compatibility,
                "issues": issues,
                "pass": '✅' if len(issues) == 0 else '⚠️'
            })
            completed_congruent.add(key)
        except Exception as error:
            errors.append({
                    "item_index": i,
                    "sentence": sentence,
                    "image": image_word,
                    "condition": "congruent",
                    "error": str(error),
                })
        save_output(congruent_items, incongruent_items, errors)
    for image_word, final_word in incongruent_pairings:
        key = f"{i}:{image_word}"
        if key in completed_incongruent:
            continue
        sentence = f"{sentence_frame} {final_word}."
        print(f"Sentence: {sentence}")
        try:
            image_path = f"{PICTURES_DIR}/{image_word}.png"
            compatibilities, explanations = rate_compatibility_with_retries(image_path, sentence_frame, final_word)
            average_compatibility, issues = check_compatibility_rating(compatibilities, explanations, 'incongruent')
            incongruent_items.append({
                "item_index": i,
                "sentence": sentence,
                "image": image_word,
                "average_compatibility": average_compatibility,
                "issues": issues,
                "pass": '✅' if len(issues) == 0 else '⚠️'
            })
            completed_incongruent.add(key)
        except Exception as error:
            errors.append({
                    "item_index": i,
                    "sentence": sentence,
                    "image": image_word,
                    "condition": "incongruent",
                    "error": str(error),
                })
        save_output(congruent_items, incongruent_items, errors)

print(f"\nProcessed {len(congruent_items) + len(incongruent_items)} items and {len(errors)} errors.")
