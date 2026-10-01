"""LoRA fine-tune of a small open model for scam detection.

A decoder LLM is adapted with LoRA to emit a short JSON verdict
(``{"label": ..., "tactics": [...]}``), which is parsed with the same code the
prompted baseline uses.  This keeps the two LLM arms comparable: same output
format, same evaluation, one prompted and one trained.

Runs on a free Colab T4 in 4-bit, and locally on Apple silicon (MPS) in bf16.

Usage::

    python -m src.models.lora --model Qwen/Qwen2.5-0.5B-Instruct --config configs/lora_qwen05b.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import load_split
from src.taxonomy import TACTICS

SYSTEM_PROMPT = "You are a fraud analyst for Indian mobile messages."


def build_prompt(tokenizer, row: dict, max_length: int) -> dict:
    answer = json.dumps(
        {"label": row["label"], "tactics": row["tactics"] if row["label"] == "scam" else []},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Message:\n\"\"\"{row['text']}\"\"\"\nLanguage: {row['language']} ({row['script']})\n"
                'Reply with JSON only: {"label": "scam"|"legit", "tactics": [<tactics>]}'
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
    labels = list(full_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100  # do not train on the prompt, only on the answer
    return {"input_ids": full_ids, "labels": labels}


class VerdictDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.items = [build_prompt(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def train(model_name: str, config: dict, device_override: str | None = None) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if device_override:
        device = device_override
    else:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.get("seed", 42))

    max_length = config.get("max_length", 256)
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 2)
    lr = config.get("lr", 1e-4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if device != "cpu" else torch.float32)
    model.config.use_cache = False
    lora = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)  # type: ignore[assignment]
    model.to(device)  # type: ignore[arg-type]
    model.print_trainable_parameters()  # type: ignore[operator]

    train_rows = load_split("train")
    val_rows = load_split("val")
    train_loader = DataLoader(
        VerdictDataset(train_rows, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        VerdictDataset(val_rows, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / (step + 1):.4f}", flush=True)
        val_loss = _val_loss(model, val_loader, device)
        print(f"epoch {epoch}: val loss {val_loss:.4f}", flush=True)

    out_dir = os.path.join("models", f"lora_{model_name.split('/')[-1]}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({**config, "model_name": model_name}, handle, indent=2)
    print(f"saved LoRA adapter to {out_dir}")


@torch.no_grad()
def _val_loss(model, loader, device) -> float:
    """Mean loss over batches that contain at least one supervised answer token.

    A row whose prompt fills ``max_length`` has every label masked to -100, and the
    cross-entropy of such a batch is NaN; those batches are skipped rather than
    poisoning the reported number.
    """
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if not (batch["labels"] != -100).any():
            continue
        batch = {key: value.to(device) for key, value in batch.items()}
        total += model(**batch).loss.item()
        count += 1
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None, help="force cpu/mps/cuda instead of auto-detection")
    args = parser.parse_args()
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
    train(args.model, config, args.device)


if __name__ == "__main__":
    main()
