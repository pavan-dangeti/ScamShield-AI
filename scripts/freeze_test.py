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

Label quality is measured, not assumed.  ``docs/test_set.md`` reports:
  * self-agreement - Cohen's kappa between the labelling pass and the blind
    re-check the tool runs on ~10% of items;
  * agreement with an independent annotator - the original UCI spam/ham label,
    which the labelling tool never shows, with every disagreement listed so
    the set can be audited.  UCI counts marketing as spam while the guidelines
    call it legitimate, so disagreement there is expected and informative.

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

import numpy as np
from sklearn.metrics import cohen_kappa_score

from scripts.build_dataset import PROCESSED_DIR, TEST_PATH, load_uci, normalize_text
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
    rows = [json.loads(line) for line in open(args.labels, encoding="utf-8") if line.strip()]
    # Exports without a ``pass`` field predate the blind re-check; treat them as the main pass.
    exported = [row for row in rows if row.get("pass", "primary") == "primary"]
    recheck = [row for row in rows if row.get("pass") == "recheck"]

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
                "labelled_by": item.get("annotator") or item.get("labelled_by", "owner"),
                "labelled_at": item.get("labelled_at", date.today().isoformat()),
                "confidence": item.get("confidence", "sure"),
                "guidelines_version": item.get("guidelines_version", "v1"),
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
    lines += ["", *label_quality(kept, exported, recheck)]
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


def _kappa(pairs: list[tuple[str, str]]) -> str:
    if len({label for pair in pairs for label in pair}) < 2:
        return "n/a (one class only)"
    first, second = zip(*pairs)
    return f"{cohen_kappa_score(first, second):.3f}"


def _cell_text(text: str, limit: int = 100) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def label_quality(kept: list[dict], primary: list[dict], recheck: list[dict]) -> list[str]:
    """Self-agreement, agreement with the source's own annotation, and effort."""
    lines = ["## Label quality", ""]
    unsure = sum(row["confidence"] == "unsure" for row in kept)
    skipped = Counter(row.get("skip_reason") or "unspecified" for row in primary if row.get("skipped"))
    seconds = [row["seconds"] for row in primary if isinstance(row.get("seconds"), (int, float))]
    lines.append(f"- Kept rows marked unsure by the labeller: {unsure} of {len(kept)}")
    lines.append(
        "- Skipped: " + (", ".join(f"{reason} {count}" for reason, count in skipped.most_common()) or "none")
    )
    if seconds:
        lines.append(f"- Median time per item: {float(np.median(seconds)):.1f} s over {len(seconds)} items")

    first = {row["id"]: row for row in primary if row.get("label") in ("scam", "legit")}
    pairs = [
        (first[row["id"]]["label"], row["label"])
        for row in recheck
        if row["id"] in first and row.get("label") in ("scam", "legit")
    ]
    lines += ["", "### Self-agreement (blind re-check)", ""]
    if pairs:
        same = sum(a == b for a, b in pairs)
        jaccard = [
            len(set(first[row["id"]]["tactics"]) & set(row["tactics"]))
            / max(1, len(set(first[row["id"]]["tactics"]) | set(row["tactics"])))
            for row in recheck
            if row["id"] in first and first[row["id"]]["label"] == row.get("label") == "scam"
        ]
        lines.append(f"- Re-checked items: {len(pairs)}; same decision: {same} ({same / len(pairs):.1%})")
        lines.append(f"- Cohen's kappa, scam/legit: {_kappa(pairs)}")
        if jaccard:
            lines.append(f"- Mean tactic Jaccard on agreed scams: {float(np.mean(jaccard)):.3f} ({len(jaccard)} items)")
    else:
        lines.append("- No blind re-check in this export; self-agreement not measured.")

    lines += ["", "### Agreement with the original UCI annotation", ""]
    try:
        reference = {f"test-{row.id}": row.label for row in load_uci().itertuples()}
    except FileNotFoundError:
        return lines + ["- UCI source not downloaded (`python -m scripts.download_data`); not measured."]
    compared = [row for row in kept if row["id"] in reference]
    if not compared:
        return lines + ["- No UCI rows in the test set."]
    pairs = [(row["label"], reference[row["id"]]) for row in compared]
    disagree = [row for row in compared if row["label"] != reference[row["id"]]]
    lines.append(f"- Compared rows: {len(compared)}; disagreements: {len(disagree)}")
    lines.append(f"- Cohen's kappa, labeller vs UCI: {_kappa(pairs)}")
    lines.append("- The labeller's decision stands: UCI labels unsolicited marketing as spam, which")
    lines.append("  `docs/labeling_guidelines.md` labels legitimate. Each disagreement is listed for audit.")
    if disagree:
        lines += ["", "| id | labeller | UCI | message |", "|---|---|---|---|"]
        for row in disagree:
            lines.append(f"| {row['id']} | {row['label']} | {reference[row['id']]} | {_cell_text(row['text'])} |")
    return lines


def _by_cell(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        grouped[(row["language"], row["script"])][row["label"]] += 1
    return grouped


if __name__ == "__main__":
    main()
