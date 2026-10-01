"""Generate LLM-written training messages with a known tactic label.

Synthetic data is for **training only**.  Every row is written to
``data/synthetic/*.jsonl`` with ``synthetic: true`` semantics (the build marks
them accordingly) and can never enter the hand-verified test set: the label
field carries the generator name and the tactics the model was asked to use,
so a reviewer can see how the row was produced.

Backends:
  * ``mlx``   - local Apple Silicon generation via mlx-lm (default, no cost)
  * ``openai`` - any OpenAI-compatible ``/v1/chat/completions`` endpoint
    (Ollama, vLLM, LM Studio, OpenAI).  Nothing is sent anywhere unless a
    base URL is given.

Run from the repository root::

    python -m scripts.generate_synthetic --backend mlx --scams-per-cell 60 --legit-per-cell 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from datetime import date

import requests

from src.pii import scrub_text
from src.taxonomy import TACTICS

SYNTHETIC_DIR = "data/synthetic"
PROMPT_VERSION = "v1"

LANGUAGE_CELLS = {
    "english": ["latin"],
    "hindi": ["native", "romanized"],
    "tamil": ["native", "romanized"],
    "telugu": ["native", "romanized"],
    "bengali": ["native", "romanized"],
}

SCRIPT_HINT = {
    "native": "written in the native script of the language",
    "romanized": (
        "written in Roman/Latin script the way people type on WhatsApp: the local language "
        "transliterated, mixed with English words such as account, click, link, update, verify"
    ),
    "latin": "written in English",
}

# Script block per language, reused from the production detector so that the
# generated corpus and the runtime language detector agree on what a script is.
LANGUAGE_BLOCK = {
    "hindi": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
}
FOREIGN_SCRIPT_BLOCKS = [(0x3000, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0x1100, 0x11FF)]

# A generated "legitimate" message must not actually be a scam.  These are the
# requests that make a message fraudulent regardless of how official it sounds.
CREDENTIAL_DEMAND = re.compile(
    r"(share|send|submit|enter|provide|confirm)\s+(your\s+)?(otp|pin|password|cvv|card details)"
    r"|verify\s+your\s+(account|kyc|identity|number)"
    r"|update\s+your\s+(kyc|account details)"
    r"|unblock\s+your\s+account"
    r"|(click|tap)[^.]{0,30}(to|and)\s+(verify|update|confirm)\s+your",
    re.IGNORECASE,
)
# Model artefacts and impossible content.
ARTEFACT = re.compile(
    r"\{\}|\[\s*\]|click here:\s*\[|fak(e|edomain)|example\.(com|net)|"
    r"\b(test|abc|xyz)\.(com|net|org|in)\b|lorem ipsum|your text here|<a href",
    re.IGNORECASE,
)

SCAM_TOPICS = [
    "fake KYC re-verification notice from a bank",
    "unclaimed prize or lottery win",
    "account or card will be blocked warning",
    "request for an OTP or UPI PIN",
    "refund or cashback claim that needs a link click",
    "electricity or gas bill disconnection threat",
    "loan or job offer that asks for an advance fee",
    "courier or customs fee notice with a link",
    "tax refund or government scheme impersonation",
    "free data pack or subscription bait",
]

LEGIT_TOPICS = [
    "an ordinary bank SMS: credit/debit confirmation, statement ready, autopay",
    "an OTP arriving from a real app login",
    "a courier delivery status update",
    "a genuine retail or bank offer with no payment link",
    "a salary or payment received notification",
    "a subscription or bill reminder asking you to pay through the official app",
    "a plain personal message with no offer at all",
    "a government scheme notice with no call to action",
]

SYSTEM_PROMPT = (
    "You write realistic short SMS and WhatsApp messages that people in India actually receive. "
    "You always answer with a single JSON object and nothing else."
)


def build_scam_prompt(language: str, script: str, topic: str) -> str:
    return (
        f"Write ONE fraudulent {language} SMS about: {topic}. "
        f"It must be {SCRIPT_HINT[script]}. "
        f"Use at least one of these tactics and list them: {', '.join(TACTICS)}. "
        "It must apply pressure or an offer, and give the victim something to do "
        "(click, call, reply, share a code). Use realistic shortened links such as bit.ly or t.ly, "
        "and never use the words fake, test or example. "
        "Reply with JSON only, in this exact shape: "
        '{"text": "<the message, under 240 characters>", "tactics": ["<one or more tactic names>"]}'
    )


def build_legit_prompt(language: str, script: str, topic: str) -> str:
    return (
        f"Write ONE legitimate, non-scam {language} SMS about: {topic}. "
        f"It must be {SCRIPT_HINT[script]}. "
        "Hard negatives: it must look like it could plausibly be read as a scam, but it must NOT "
        "ask for an OTP, PIN, password or card details, must NOT use a countdown or threat, and "
        "must NOT contain a shortened or misspelled URL. Payment happens in the official app. "
        "Reply with JSON only, in this exact shape: "
        '{"text": "<the message, under 240 characters>", "tactics": []}'
    )


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).strip().lower()


def parse_response(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload.get("text"), str):
        return None
    return payload


class MlxBackend:
    def __init__(self, model: str, temperature: float = 0.9) -> None:
        from mlx_lm import generate, load  # imported lazily: macOS/Apple silicon only
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self._sampler = make_sampler(temp=temperature, top_p=0.95)
        # Tolerate both the 2-tuple and (if return_config ever defaults on) 3-tuple.
        loaded = load(model)
        self.model, self.tokenizer = loaded[0], loaded[1]

    def generate(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        templated = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._generate(self.model, self.tokenizer, templated, max_tokens=220, sampler=self._sampler)


class OpenAiBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "") -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 1.0,
                "max_tokens": 220,
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def script_compliance(text: str, language: str, script: str) -> bool:
    """Reject rows written in the wrong script or contaminated by another one."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    own = LANGUAGE_BLOCK.get(language)
    own_fraction = (
        sum(1 for character in letters if own[0] <= ord(character) <= own[1]) / len(letters)
        if own
        else 0.0
    )
    foreign = sum(
        1
        for character in letters
        for start, end in FOREIGN_SCRIPT_BLOCKS
        if start <= ord(character) <= end
    ) / len(letters)
    if foreign > 0.01:
        return False
    if script == "native":
        return own_fraction >= 0.5
    if script == "romanized":
        return own_fraction <= 0.1
    return own_fraction == 0.0 and foreign == 0.0


def validate(payload: dict, language: str, script: str, label: str, seen: set[str]) -> dict | None:
    text = scrub_text(payload["text"])
    if not 10 <= len(text) <= 400:
        return None
    if normalize(text) in seen:
        return None
    if not script_compliance(text, language, script):
        return None
    if ARTEFACT.search(text):
        return None
    if label == "legit" and CREDENTIAL_DEMAND.search(text):
        return None  # generated as legitimate but actually demands credentials
    tactics = [t for t in payload.get("tactics", []) if t in TACTICS]
    if label == "scam" and not tactics:
        return None
    if label == "legit" and tactics:
        return None
    seen.add(normalize(text))
    return {"text": text, "tactics": tactics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--scams-per-cell", type=int, default=60)
    parser.add_argument("--legit-per-cell", type=int, default=30)
    parser.add_argument(
        "--languages",
        default="all",
        help="comma separated subset of " + ",".join(LANGUAGE_CELLS) + " (default: all)",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(SYNTHETIC_DIR, "generated.jsonl"))
    args = parser.parse_args()

    wanted = LANGUAGE_CELLS if args.languages == "all" else {
        name: LANGUAGE_CELLS[name] for name in args.languages.split(",") if name in LANGUAGE_CELLS
    }
    if not wanted:
        raise SystemExit(f"--languages must name known languages: {', '.join(LANGUAGE_CELLS)}")

    backend = MlxBackend(args.model) if args.backend == "mlx" else OpenAiBackend(args.base_url, args.model, args.api_key)

    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    seen: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.add(normalize(json.loads(line)["text"]))

    written = rejected = 0
    started = time.time()
    with open(args.out, "a", encoding="utf-8") as handle:
        for language, scripts in wanted.items():
            for script in scripts:
                jobs = [("scam", topic) for topic in SCAM_TOPICS for _ in range(args.scams_per_cell // len(SCAM_TOPICS) + 1)]
                jobs += [("legit", topic) for topic in LEGIT_TOPICS for _ in range(args.legit_per_cell // len(LEGIT_TOPICS) + 1)]
                for label, topic in jobs:
                    for _ in range(args.max_attempts):
                        prompt = (build_scam_prompt if label == "scam" else build_legit_prompt)(
                            language, script, topic
                        )
                        try:
                            raw = backend.generate(prompt)
                        except Exception as error:  # noqa: BLE001 - keep the batch alive
                            print(f"  generation error: {error}")
                            break
                        payload = parse_response(raw)
                        record = validate(payload, language, script, label, seen) if payload else None
                        if record is None:
                            continue
                        handle.write(
                            json.dumps(
                                {
                                    "id": f"gen-{language}-{script}-{label}-{written:05d}",
                                    "text": record["text"],
                                    "language": language,
                                    "script": script,
                                    "label": label,
                                    "tactics": record["tactics"],
                                    "generator": f"{args.backend}:{args.model}",
                                    "prompt_version": PROMPT_VERSION,
                                    "created": date.today().isoformat(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                        written += 1
                        break
                    else:
                        rejected += 1
                print(f"{language}/{script}: running total {written} kept, {rejected} rejected")
    print(f"Done: {written} new rows, {rejected} rejected, {time.time() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
