"""Prompted LLM baselines: zero-shot and few-shot scam detection.

The model is asked for a JSON verdict so that the tactic list can be scored the
same way as the trained models.  Backends:

* ``mlx``     - local Apple silicon generation (default), e.g. Qwen2.5-7B-Instruct-4bit
* ``openai``  - any OpenAI-compatible /v1/chat/completions endpoint

Both are free.  An API model can be pointed at with ``--backend openai``; the
scripts never call a hosted service unless an endpoint is passed explicitly.

Usage::

    python -m src.models.llm_prompted --shots 0 --limit 400
    python -m src.models.llm_prompted --shots 3 --limit 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import numpy as np

from src.taxonomy import TACTICS

SYSTEM_PROMPT = (
    "You are a fraud analyst for Indian mobile messages. You read SMS and WhatsApp messages "
    "in English and in code-mixed Indian languages and decide whether a message is a scam."
)

GUIDELINES = """Decide using these rules:
- scam if the message would cost the reader money, credentials or account access if acted on.
- Tactics you may report: urgency, authority_impersonation, false_reward, loss_aversion,
  credential_phishing, suspicious_link. Report every tactic genuinely present, and none that is absent.
- A bank telling the reader not to share an OTP is legitimate. An OTP arriving by SMS is legitimate.
- A discount or offer is legitimate unless it demands a fee, a code, or a click on an unknown link.
- Marketing that is merely pushy is not a scam unless it is deceptive."""

USER_TEMPLATE = """Message:
\"\"\"{text}\"\"\"

Language: {language} ({script})

Reply with JSON only:
{{"label": "scam" or "legit", "tactics": [<subset of the tactic list>]}}"""


def few_shot_examples(rows: list[dict], count: int, seed: int) -> str:
    """Class-balanced few-shot examples drawn from the training split."""
    rng = random.Random(seed)
    scams = [row for row in rows if row["label"] == "scam" and row["tactics"]]
    legits = [row for row in rows if row["label"] == "legit"]
    picked: list[dict] = []
    for group in (scams, legits):
        picked.extend(rng.sample(group, min(len(group), count // 2 + count % 2)))
    rng.shuffle(picked)
    blocks = []
    for row in picked:
        answer = {"label": row["label"], "tactics": row["tactics"]}
        blocks.append(
            f"Message:\n\"\"\"{row['text']}\"\"\"\nLanguage: {row['language']} ({row['script']})\n"
            f"Answer: {json.dumps(answer, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


class MlxChat:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self._sampler = make_sampler(temp=temperature, top_p=1.0) if temperature else make_sampler(temp=0.0)
        self.model, self.tokenizer = load(model)

    def __call__(self, system: str, user: str, max_tokens: int = 160) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=self._sampler)


class OpenAiChat:
    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.0) -> None:
        import requests

        self._requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def __call__(self, system: str, user: str, max_tokens: int = 160) -> str:
        response = self._requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def parse_verdict(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    label = str(payload.get("label", "")).strip().lower()
    if label not in ("scam", "legit"):
        return None
    tactics = [t for t in payload.get("tactics", []) if t in TACTICS]
    if label == "scam" and not tactics:
        tactics = ["urgency"]  # a scam must carry at least one tactic
    return {"label": label, "tactics": tactics if label == "scam" else []}


def run(rows: list[dict], chat, shots: int, train_rows: list[dict], seed: int) -> list[dict]:
    prefix = few_shot_examples(train_rows, shots, seed) if shots else ""
    system = SYSTEM_PROMPT if not prefix else f"{SYSTEM_PROMPT}\n\n{GUIDELINES}\n\nExamples:\n{prefix}"
    results = []
    for index, row in enumerate(rows, 1):
        prompt = USER_TEMPLATE.format(text=row["text"], language=row["language"], script=row["script"])
        verdict = parse_verdict(chat(system, prompt))
        if verdict is None:
            verdict = {"label": "legit", "tactics": []}  # unparsable output is treated as no detection
        results.append({"id": row["id"], **verdict, "raw": None})
        if index % 50 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=400, help="stratified subsample of the split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from src.data import load_split, subsample_rows

    rows = subsample_rows(load_split(args.split), args.limit, args.seed)
    print(f"{len(rows)} rows from {args.split}, {args.shots}-shot, model={args.model}")

    chat = MlxChat(args.model) if args.backend == "mlx" else OpenAiChat(args.base_url, args.model, args.api_key)
    results = run(rows, chat, args.shots, load_split("train"), args.seed)

    out = args.out or f"results/llm_{args.shots}shot_{args.split}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} verdicts to {out}")


if __name__ == "__main__":
    main()
