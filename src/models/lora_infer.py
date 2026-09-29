"""Inference wrapper for the LoRA-tuned decoder model."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.models.llm_prompted import SYSTEM_PROMPT, parse_verdict


def load_lora(model_name: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = os.path.join("models", f"lora_{model_name}")
    # The adapter config records the exact base checkpoint it was trained against.
    with open(os.path.join(directory, "adapter_config.json"), encoding="utf-8") as handle:
        base = json.load(handle)["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, directory)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer, device


@torch.no_grad()
def predict_lora(model_name: str, texts: list[str], rows: list[dict], max_new_tokens: int = 48):
    from src.models.lora import SYSTEM_PROMPT as LORA_SYSTEM

    model, tokenizer, device = load_lora(model_name)
    probabilities, tactics = [], []
    for index, row in enumerate(rows):
        user = (
            f"Message:\n\"\"\"{row['text']}\"\"\"\nLanguage: {row['language']} ({row['script']})\n"
            'Reply with JSON only: {"label": "scam"|"legit", "tactics": [<tactics>]}'
        )
        messages = [
            {"role": "system", "content": LORA_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = parse_verdict(text)
        if verdict is None:
            probabilities.append(0.0)
            tactics.append([])
            continue
        probabilities.append(1.0 if verdict["label"] == "scam" else 0.0)
        tactics.append(verdict["tactics"])
    return np.array(probabilities), tactics
