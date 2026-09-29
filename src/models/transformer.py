"""Fine-tune a multilingual encoder (MuRIL, XLM-R) for scam detection and
tactic classification.

One shared backbone, two heads: a binary head for scam/legit and a 6-way
multi-label head for the manipulation tactics.  That mirrors the baseline's two
heads so the two model families are compared like for like.

Designed to run on a free Colab/Kaggle T4.  On Apple silicon it also runs on
MPS, which is how the committed numbers were produced.

Usage::

    python -m src.models.transformer --model google/muril-base --config configs/muril.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import load_split
from src.taxonomy import TACTICS

ENCODER_DEFAULTS = {
    "google/muril-base": 128,
    "ai4bharat/IndicBERTv2-MLM-only": 128,
    "xlm-roberta-base": 160,
    "xlm-roberta-large": 192,
}


class ScamDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(1 + len(TACTICS), dtype=torch.float32)
        labels[0] = 1.0 if row["label"] == "scam" else 0.0
        for i, tactic in enumerate(TACTICS, start=1):
            labels[i] = 1.0 if tactic in row.get("tactics", []) else 0.0
        encoded["labels"] = labels.squeeze(0)
        return {key: value.squeeze(0) for key, value in encoded.items()}


class ScamEncoderModel(torch.nn.Module):
    """Encoder + binary head + multi-label tactic head."""

    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.binary_head = torch.nn.Linear(hidden, 1)
        self.tactic_head = torch.nn.Linear(hidden, len(TACTICS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            outputs = self.encoder(**kwargs)
        except TypeError:
            # XLM-R has no token_type_ids; MuRIL (BERT) does.
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS]
        return self.binary_head(pooled).squeeze(-1), self.tactic_head(pooled)


def _batch_forward(model, batch, device):
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("input_ids", "attention_mask", "token_type_ids")
    }
    return model(**inputs)


def train(model_name: str, config: dict, train_split: str = "train", output_suffix: str = "") -> None:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.get("seed", 42))

    max_length = config.get("max_length", ENCODER_DEFAULTS.get(model_name, 160))
    batch_size = config.get("batch_size", 16 if device != "cpu" else 8)
    epochs = config.get("epochs", 3)
    lr = config.get("lr", 2e-5)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_rows = load_split(train_split)
    val_rows = load_split("val")
    train_dataset = ScamDataset(train_rows, tokenizer, max_length)
    val_dataset = ScamDataset(val_rows, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = ScamEncoderModel(model_name, dropout=config.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.01))
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    tactic_pos_weight = _tactic_weights(train_rows, device)
    binary_criterion = torch.nn.BCEWithLogitsLoss()
    tactic_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=tactic_pos_weight)

    best_val_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            binary_logits, tactic_logits = model(**inputs)
            binary_labels = batch["labels"][:, 0].to(device)
            tactic_labels = batch["labels"][:, 1:].to(device)
            loss = binary_criterion(binary_logits, binary_labels) + 0.5 * tactic_criterion(
                tactic_logits, tactic_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / (step + 1):.4f}", flush=True)
        val_metrics = _evaluate(model, val_loader, device, config)
        print(f"epoch {epoch}: val {val_metrics}", flush=True)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_dir = os.path.join("models", model_name.replace("/", "_") + output_suffix)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {**config, "model_name": model_name, "max_length": max_length,
             "train_split": train_split, "val_macro_f1": best_val_f1},
            handle, indent=2,
        )
    print(f"saved to {save_dir} (val macro F1 {best_val_f1:.4f})")


def _tactic_weights(rows: list[dict], device) -> torch.Tensor:
    counts = np.array([sum(1 for r in rows if t in r.get("tactics", [])) for t in TACTICS], dtype=np.float64)
    negatives = len(rows) - counts
    weight = np.where(counts > 0, negatives / np.maximum(counts, 1), 1.0)
    weight = np.clip(weight, 1.0, 20.0)
    return torch.tensor(weight, dtype=torch.float32, device=device)


@torch.no_grad()
def _evaluate(model, loader, device, config) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    model.eval()
    binary_true, binary_pred = [], []
    tactic_true, tactic_pred = [], []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        binary_logits, tactic_logits = model(**inputs)
        binary_true.extend(batch["labels"][:, 0].numpy().tolist())
        binary_pred.extend((torch.sigmoid(binary_logits) >= 0.5).float().cpu().numpy().tolist())
        tactic_true.extend(batch["labels"][:, 1:].numpy())
        tactic_pred.extend((torch.sigmoid(tactic_logits) >= 0.5).float().cpu().numpy())
    tactic_true = np.array(tactic_true)
    tactic_pred = np.array(tactic_pred)
    return {
        "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(tactic_true, tactic_pred, average="macro", zero_division=0)
        ),
        "tactic_micro_f1": float(
            f1_score(tactic_true, tactic_pred, average="micro", zero_division=0)
        ),
        "config": config.get("name", "default"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/muril-base")
    parser.add_argument("--config", default=None, help="path to a JSON training config")
    parser.add_argument("--train-split", default="train", help="which processed split to train on")
    parser.add_argument("--output-suffix", default="", help="appended to the model directory, e.g. -aug")
    args = parser.parse_args()
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
    train(args.model, config, args.train_split, args.output_suffix)


if __name__ == "__main__":
    main()
