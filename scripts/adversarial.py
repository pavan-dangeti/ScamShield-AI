"""Adversarial perturbations for scam messages.

Every perturbation is deterministic (seeded) and applied to the text only, never to
the label, so a perturbed row keeps the gold answer. The six families are the ones a
sender uses to slip a scam past a filter:

* ``char_swap``      - leetspeak substitutions (a->@, i->1, o->0 ...)
* ``lookalike``      - Cyrillic/Greek/fullwidth characters that render like Latin
* ``transliteration``- romanised spelling drift ("click" -> "clik", "karein" -> "kerein")
* ``spacing``        - zero-width spaces, doubled spaces, spaces inside words
* ``emoji``          - emoji injected into the sentence
* ``disguised_link`` - link tricks (bit[.]ly, "b i t . l y", "hxxps")

Usage::

    python -m scripts.adversarial --dump data/processed/val_adv.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata

FAMILIES = ["char_swap", "lookalike", "transliteration", "spacing", "emoji", "disguised_link"]

# Cyrillic and Greek letters that render identically to Latin ones.
LOOKALIKE = {
    "a": "а",  # CYRILLIC SMALL A
    "c": "с",  # CYRILLIC SMALL ES
    "e": "е",  # CYRILLIC SMALL IE
    "o": "о",  # CYRILLIC SMALL O
    "p": "р",  # CYRILLIC SMALL ER
    "x": "х",  # CYRILLIC SMALL HA
    "y": "у",  # CYRILLIC SMALL U
    "i": "і",  # CYRILLIC SMALL BYELORUSSIAN-UKRAINIAN I
    "s": "ѕ",  # CYRILLIC SMALL DZE
    "A": "А",
    "E": "Е",
    "O": "О",
    "P": "Р",
    "C": "С",
    "T": "Т",
    "H": "Н",
    "M": "М",
    "B": "В",
    "K": "К",
}

LEET = {"a": "@", "i": "1", "o": "0", "e": "3", "s": "$", "l": "1", "t": "7", "b": "8", "g": "9"}

DRIFT = [
    (re.compile(r"\bclick\b", re.I), "clik"),
    (re.compile(r"\bblock\b", re.I), "blok"),
    (re.compile(r"\bkarein\b", re.I), "kerein"),
    (re.compile(r"\bkaro\b", re.I), "kro"),
    (re.compile(r"\bjaldi\b", re.I), "jldi"),
    (re.compile(r"\baccount\b", re.I), "acount"),
    (re.compile(r"\bverify\b", re.I), "verfy"),
    (re.compile(r"\baccount\b", re.I), "accunt"),
    (re.compile(r"\bimmediately\b", re.I), "immediatly"),
    (re.compile(r"\bpayment\b", re.I), "paymnt"),
    (re.compile(r"\brefund\b", re.I), "refnd"),
    (re.compile(r"\bplease\b", re.I), "plese"),
]

EMOJI = [
    "\U0001F4B3",  # money bag
    "\U0001F680",  # rocket
    "⚠️",  # warning
    "\U0001F512",  # lock
    "\U0001F4E6",  # package
    "\U0001F440",  # eyes
]

ZERO_WIDTH = "\u200b"  # zero-width space


def _letters(text: str) -> list[int]:
    return [index for index, character in enumerate(text) if character.isalpha() and character.isascii()]


def char_swap(text: str, rng: random.Random, rate: float = 0.3) -> str:
    indices = [index for index, character in enumerate(text) if character in LEET]
    if not indices:
        return text
    chosen = rng.sample(indices, k=max(1, int(len(indices) * rate)))
    return "".join(LEET.get(text[index], text[index]) if index in chosen else character
                   for index, character in enumerate(text))


def lookalike(text: str, rng: random.Random, rate: float = 0.3) -> str:
    indices = [index for index, character in enumerate(text) if character in LOOKALIKE]
    if not indices:
        return text
    chosen = set(rng.sample(indices, k=max(1, int(len(indices) * rate))))
    return "".join(LOOKALIKE.get(character, character) if index in chosen else character
                   for index, character in enumerate(text))


def transliteration(text: str, rng: random.Random, rate: float = 0.4) -> str:
    result = text
    applied = 0
    for pattern, replacement in DRIFT:
        if rng.random() < rate:
            result, count = pattern.subn(replacement, result)
            applied += count
    if applied:
        return result
    # No known word: drift an arbitrary letter by doubling it.
    indices = _letters(text)
    if not indices:
        return text
    index = rng.choice(indices)
    return text[: index + 1] + text[index] + text[index:]


def spacing(text: str, rng: random.Random, rate: float = 0.2) -> str:
    mode = rng.choice(["zero_width", "wide_space", "letter_space"])
    indices = _letters(text)
    if not indices:
        return text
    if mode == "zero_width":
        chosen = set(rng.sample(indices, k=max(1, int(len(indices) * rate))))
        return "".join(character + (ZERO_WIDTH if index in chosen else "") for index, character in enumerate(text))
    if mode == "wide_space":
        chosen = set(rng.sample(indices, k=max(1, int(len(indices) * rate))))
        return "".join(character + (" " if index in chosen else "") for index, character in enumerate(text))
    words = text.split(" ")
    if len(words) < 2:
        return text + " " + ZERO_WIDTH
    return "  ".join(words)


def emoji(text: str, rng: random.Random, count: int = 2) -> str:
    symbols = rng.sample(EMOJI, k=min(count, len(EMOJI)))
    words = text.split(" ")
    if len(words) < 2:
        return " ".join(symbols) + " " + text
    positions = sorted(rng.sample(range(len(words)), k=min(count, len(words))))
    for position in positions:
        words[position] = f"{words[position]} {symbols.pop(0)}"
    return " ".join(word for word in words if word)


def disguised_link(text: str, rng: random.Random) -> str:
    """Obfuscate a URL the way a filter-evading sender would."""
    pattern = re.compile(r"(https?://|www\.)?[\w-]+(\.[\w-]+)+(/[^\s]*)?", re.IGNORECASE)

    def replace(match: re.Match) -> str:
        url = match.group(0)
        mode = rng.choice(["dots", "space", "scheme", "typo", "at"])
        if mode == "dots":
            return url.replace(".", "[.]")
        if mode == "space":
            return re.sub(r"([a-z])", r"\1 ", url, count=3)
        if mode == "scheme":
            return url.replace("https://", "hxxps://").replace("http://", "hxxp://")
        if mode == "typo":
            return url.replace("http", "hittp").replace(".com", ".cpm")
        return url.replace("http", "httpⓢ").replace("://", "://ⓢ")

    # A message with no link is left alone: appending a fake link to a legitimate
    # bank notification would change its meaning, not just its surface form.
    if not pattern.search(text):
        return text
    return pattern.sub(replace, text, count=1)


PERTURBATIONS = {
    "char_swap": char_swap,
    "lookalike": lookalike,
    "transliteration": transliteration,
    "spacing": spacing,
    "emoji": emoji,
    "disguised_link": disguised_link,
}


def perturb(text: str, family: str, seed: int) -> str:
    """Apply one family deterministically. Returns the text unchanged if it cannot apply."""
    rng = random.Random(f"{seed}:{family}:{text[:24]}")
    try:
        perturbed = PERTURBATIONS[family](text, rng)
    except Exception:
        return text
    # A perturbation that produces nothing (e.g. no ASCII letters in native script)
    # would silently weaken the test, so those rows are marked as unperturbed.
    return unicodedata.normalize("NFC", perturbed)


def build_adversarial_split(rows: list[dict], family: str, seed: int = 42) -> tuple[list[dict], int]:
    """Return (perturbed rows, how many rows the perturbation could not change)."""
    out, unchanged = [], 0
    for row in rows:
        perturbed = perturb(row["text"], family, seed)
        if perturbed == row["text"]:
            unchanged += 1
        out.append({**row, "text": perturbed, "attack": family})
    return out, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--family", default="all", choices=["all", *FAMILIES])
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    from src.data import load_split

    rows = load_split(args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    for family in FAMILIES if args.family == "all" else [args.family]:
        perturbed, unchanged = build_adversarial_split(rows, family)
        path = f"{args.out_dir}/{args.split}_adv_{family}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in perturbed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{family}: {len(perturbed)} rows -> {path} ({unchanged} unchanged by the perturbation)")


if __name__ == "__main__":
    import os  # noqa: E402  (used by main only)

    main()
