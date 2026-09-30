"""Build an adversarially augmented training set.

Takes the clean training split and adds perturbed copies of a fraction of its rows,
keeping the original label and tactic labels. The perturbation families are the same
ones used for evaluation, so a model trained on this set has seen the attack shapes
before - which is the point of augmentation, not a way to leak the test signal: the
copies come from *training* rows only, and the validation split stays clean.

Usage::

    python -m scripts.build_augmented_dataset --fraction 0.5 --families char_swap lookalike spacing
"""

from __future__ import annotations

import argparse
import json
import os
import random

from scripts.adversarial import FAMILIES, perturb
from src.data import load_split

PROCESSED_DIR = "data/processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.5, help="copies per training row, e.g. 0.5")
    parser.add_argument("--families", nargs="+", default=["char_swap", "lookalike", "spacing", "disguised_link"], choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(PROCESSED_DIR, "train_augmented.jsonl"))
    args = parser.parse_args()

    rows = load_split("train")
    rng = random.Random(args.seed)
    augmented, skipped = [], 0
    for row in rows:
        copies = int(args.fraction)
        remainder = args.fraction - copies
        if rng.random() < remainder:
            copies += 1
        for index in range(copies):
            family = rng.choice(args.families)
            text = perturb(row["text"], family, args.seed + index)
            if text == row["text"]:
                skipped += 1
                continue
            augmented.append({**row, "id": f"{row['id']}-aug{index}-{family}", "text": text, "augmented_from": row["id"], "attack": family})

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in augmented:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(rows)} clean + {len(augmented)} augmented rows "
        f"({skipped} perturbations produced no change and were skipped)"
    )
    by_family = {}
    for row in augmented:
        by_family[row["attack"]] = by_family.get(row["attack"], 0) + 1
    print("  by family:", by_family)


if __name__ == "__main__":
    main()
