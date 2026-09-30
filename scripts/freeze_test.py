"""Freeze the hand-verified test set from the label export produced by the
labelling tool (docs/labeling/labeler.html).

Hard requirements (the script fails if any of them is broken):
  * every kept row has label scam or legit
  * every kept row has provenance "real" - no synthetic, no unverified source
  * no duplicate messages inside the test set
  * at least one label per (language, script, label) cell

Soft requirement: ``--min-per-cell`` (default 25) per cell and class.  Cells
below the target are reported as a gap in ``docs/test_set.md`` rather than
silently accepted.

After freezing, re-run ``python -m scripts.build_dataset``; the build excludes
every training row that is a near-duplicate of a frozen test message.

Usage::

    python -m scripts.freeze_test --labels data/test_candidates/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date

from scripts.build_dataset import PROCESSED_DIR, TEST_PATH, normalize_text
from scripts.make_test_candidates import CANDIDATES
from src.taxonomy import TACTICS

TEST_SET_DOC = "docs/test_set.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/test_candidates/labels.jsonl")
    parser.add_argument("--min-per-cell", type=int, default=25)
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        raise SystemExit(f"No label export at {args.labels}; run the labelling tool first.")
    if not os.path.exists(CANDIDATES):
        raise SystemExit("No candidate pool; run scripts.make_test_candidates first.")

    candidates = {
        json.loads(line)["id"]: json.loads(line)
        for line in open(CANDIDATES, encoding="utf-8")
        if line.strip()
    }
    exported = [json.loads(line) for line in open(args.labels, encoding="utf-8") if line.strip()]

    kept: list[dict] = []
    problems: list[str] = []
    seen: set[str] = set()
    for item in exported:
        if item.get("skipped") or item.get("label") not in ("scam", "legit"):
            continue
        candidate = candidates.get(item["id"])
        if candidate is None:
            problems.append(f"{item['id']}: not in candidate pool (dropped)")
            continue
        if candidate["provenance"] != "real" or candidate.get("synthetic"):
            problems.append(f"{item['id']}: provenance {candidate['provenance']} is not real (dropped)")
            continue
        if candidate["text"] != item["text"]:
            problems.append(f"{item['id']}: text changed after labelling (dropped)")
            continue
        key = normalize_text(candidate["text"])
        if key in seen:
            problems.append(f"{item['id']}: duplicate inside the test set (dropped)")
            continue
        seen.add(key)
        tactics = [t for t in item.get("tactics", []) if t in TACTICS]
        if item["label"] == "scam" and not tactics:
            problems.append(f"{item['id']}: scam without tactics (dropped)")
            continue
        kept.append(
            {
                "id": candidate["id"],
                "text": candidate["text"],
                "language": candidate["language"],
                "script": candidate["script"],
                "label": item["label"],
                "tactics": tactics if item["label"] == "scam" else [],
                "source": candidate["source"],
                "source_id": candidate.get("source_id", ""),
                "licence": candidate.get("licence", "project"),
                "provenance": "real",
                "synthetic": False,
                "split": "test",
                "labelled_by": item.get("labelled_by", "owner"),
                "labelled_at": item.get("labelled_at", date.today().isoformat()),
            }
        )

    cells: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in kept:
        cells[(row["language"], row["script"], row["label"])] += 1
    for (language, script, label), count in sorted(cells.items()):
        if count < args.min_per_cell:
            problems.append(
                f"cell {language}/{script}/{label}: {count} rows, below target {args.min_per_cell}"
            )
    for language in ("english", "hindi", "tamil", "telugu", "bengali"):
        if not any(key[0] == language for key in cells):
            problems.append(f"no test rows at all for {language}")

    if not kept:
        raise SystemExit("No usable labels. Nothing frozen.\n" + "\n".join(problems))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(TEST_PATH, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Hand-verified test set",
        "",
        f"Frozen on {date.today().isoformat()} from {args.labels}.",
        "Rules: real messages only, no synthetic or unverified-provenance rows, no duplicates,",
        "scam rows carry at least one tactic, and the build excludes near-duplicates of these",
        "rows from training. Labelling rules: `docs/labeling_guidelines.md`.",
        "",
        f"Total rows: **{len(kept)}**",
        "",
        "| Language | Script | Legit | Scam |",
        "|---|---|---|---|",
    ]
    for (language, script), group in sorted(_by_cell(kept).items()):
        lines.append(
            f"| {language} | {script} | {group.get('legit', 0)} | {group.get('scam', 0)} |"
        )
    lines += ["", "## Gaps", ""]
    lines += [f"- {problem}" for problem in problems] or ["- none"]
    lines += ["", "## Sources", ""]
    for source, count in Counter(row["source"] for row in kept).most_common():
        lines.append(f"- {source}: {count}")
    with open(TEST_SET_DOC, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Froze {len(kept)} rows -> {TEST_PATH} and {TEST_SET_DOC}")
    for problem in problems:
        print(f"  note: {problem}")


def _by_cell(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        grouped[(row["language"], row["script"])][row["label"]] += 1
    return grouped


if __name__ == "__main__":
    main()
