"""Import real messages the owner collected, for the hand-verified test set.

This is the only way to get real Hindi/Tamil/Telugu (and more Bengali) coverage
into the test set: no licence-clean public corpus exists for them.  Read the CSV
before sharing anything: **remove names, your own phone number, account numbers
and anything else identifying from the text**.  The importer scrubs phone
numbers, emails, UPI handles and link query strings automatically, but it cannot
recognise personal names.

CSV columns (only ``text`` is required)::

    text,language,script
    "Unga account 30 nimishathula block aagidum, ippo click pannunga: bit.ly/xyz",tamil,romanized

``language`` must be one of english, hindi, tamil, telugu, bengali.  ``script``
is one of native, romanized, latin and is inferred with the production
``ScriptLanguageDetector`` when omitted.

Usage::

    python -m scripts.import_user_messages --input my_messages.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from scripts.build_dataset import normalize_text
from scripts.make_test_candidates import CANDIDATE_DIR, USER_IMPORT
from src.detector import ScriptLanguageDetector
from src.pii import scrub_text

LANGUAGES = ("english", "hindi", "tamil", "telugu", "bengali")
SCRIPTS = ("native", "romanized", "latin")


def detect_row(text: str, detector: ScriptLanguageDetector) -> tuple[str, str]:
    detected = detector.detect(text)
    script = detected["script_type"]
    language = detected["language_guess"].lower()
    if script == "latin":
        script = "latin"
    if language not in LANGUAGES:
        raise ValueError(
            f"could not attribute message to a supported language (got '{language}'). "
            "Set the language and script columns explicitly."
        )
    return language, script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV file with a text column")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "text" not in frame.columns:
        raise SystemExit("CSV must have a 'text' column")
    detector = ScriptLanguageDetector("data/language_packs")

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    existing: set[str] = set()
    rows: list[dict] = []
    if os.path.exists(USER_IMPORT):
        with open(USER_IMPORT, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.add(normalize_text(json.loads(line)["text"]))

    skipped = 0
    for _, row in frame.iterrows():
        text = scrub_text(str(row["text"]))
        if len(text) < 10:
            skipped += 1
            continue
        if normalize_text(text) in existing:
            skipped += 1
            continue
        language = str(row.get("language") or "").strip().lower()
        script = str(row.get("script") or "").strip().lower()
        try:
            if not language:
                language, script = detect_row(text, detector)
            elif not script:
                language, script = detect_row(text, detector)
            if language not in LANGUAGES:
                raise ValueError(f"unsupported language '{language}'")
            if script not in SCRIPTS:
                raise ValueError(f"unsupported script '{script}'")
        except ValueError as error:
            print(f"skipped: {error} :: {text[:60]}", file=sys.stderr)
            skipped += 1
            continue
        existing.add(normalize_text(text))
        rows.append({"id": f"user-{len(rows):05d}", "text": text, "language": language, "script": script})

    with open(USER_IMPORT, "a", encoding="utf-8") as handle:
        for message in rows:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    counts = pd.Series([f"{row['language']}|{row['script']}" for row in rows]).value_counts().to_dict()
    print(f"Imported {len(rows)} messages into {USER_IMPORT} ({skipped} skipped)")
    for cell, count in sorted(counts.items()):
        print(f"  {cell}: {count}")


if __name__ == "__main__":
    main()
