"""Build the ScamShield training/validation dataset from licensed sources.

Pipeline: gather -> scrub PII -> exact dedup -> near-duplicate grouping ->
group-aware stratified train/val split.  Rows from the frozen hand-labelled
test set (``data/processed/test.jsonl``) are excluded from training, including
every near-duplicate of a test message.

Run from the repository root::

    python -m scripts.build_dataset
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import tempfile
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scripts.scrub_pii import scrub_text
from src.taxonomy import TACTICS

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SYNTHETIC_DIR = "data/synthetic"
TEST_PATH = os.path.join(PROCESSED_DIR, "test.jsonl")

# Near-duplicate grouping.  The metric is deliberately simple and corpus
# independent of word boundaries: raw character 4-5 gram TF (no IDF), cosine
# similarity.  Groups are always kept inside a single split.
# 0.82 is calibrated so that scripts/check_leakage.py, which recomputes the
# same metric on the written split files, finds no cross-split pair above 0.90.
NEAR_DUP_VECTORIZER = {
    "analyzer": "char",
    "ngram_range": (4, 5),
    "min_df": 1,
    "use_idf": False,
}
NEAR_DUP_THRESHOLD = 0.82

SOURCE_PRIORITY = {
    "uci_sms_spam": 0,
    "bengali_sms_smishing": 1,
    "indian_scam_sms_synthetic": 2,
    "template_pack": 3,
    "generated_synthetic": 4,
}

# The only licence-clean real corpus is the UCI SMS Spam Collection, so it is
# partitioned before anything is trained on it: this many scam and ham messages
# are held out as an unlabelled reserve for the hand-verified test set and never
# enter the training pool, labelled or not.  The rest is training material.
UCI_RESERVE = {"scam": 120, "legit": 180}
RESERVE_HINT = "test_reserve"

RIDHAM_LANGUAGE_MAP = {
    "en": ("english", "latin"),
    "hi": ("hindi", "native"),
    "hinglish": ("hindi", "romanized"),
    "ta_en": ("tamil", "romanized"),
    "te_en": ("telugu", "romanized"),
    "bn_en": ("bengali", "romanized"),
}

BENGALI_VARIETY_MAP = {
    "Bengali": ("bengali", "native"),
    "Banglish": ("bengali", "romanized"),
    "CodeMix": ("bengali", "romanized"),
    "English": ("english", "latin"),
}

SCHEMA_FIELDS = [
    "id",
    "text",
    "language",
    "script",
    "label",
    "tactics",
    "source",
    "source_id",
    "licence",
    "provenance",
    "synthetic",
    "split",
]


def normalize_text(text: str) -> str:
    """Unicode-normalise and collapse whitespace for comparison."""
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.split()).strip()


def make_id(source: str, text: str) -> str:
    return f"{source}-{hashlib.sha1(normalize_text(text).encode('utf-8')).hexdigest()[:12]}"


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=SCHEMA_FIELDS)
    frame["text"] = frame["text"].map(scrub_text)
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    return frame


def load_uci() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "uci_sms_spam", "SMSSpamCollection")
    raw = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    label_map = {"spam": "scam", "ham": "legit"}
    rows = [
        {
            "id": make_id("uci_sms_spam", text),
            "text": text,
            "language": "english",
            "script": "latin",
            "label": label_map[label],
            "tactics": [],
            "source": "uci_sms_spam",
            "source_id": str(index),
            "licence": "CC-BY-4.0",
            "provenance": "real",
            "synthetic": False,
            "split": None,
        }
        for index, (label, text) in enumerate(zip(raw["label"], raw["text"]))
    ]
    frame = _frame(rows)
    frame["split_hint"] = "pool"
    for label, count in UCI_RESERVE.items():
        chosen = frame[frame["label"] == label].sample(n=count, random_state=42).index
        frame.loc[chosen, "split_hint"] = RESERVE_HINT
    return frame


def load_ridham() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "indian_scam_sms_synthetic", "synthetic_audited.csv")
    raw = pd.read_csv(path)
    label_map = {"scam": "scam", "safe": "legit"}
    rows = []
    for _, row in raw.iterrows():
        mapping = RIDHAM_LANGUAGE_MAP.get(row["language"])
        if mapping is None:
            continue  # out of scope (e.g. mr_en)
        language, script = mapping
        rows.append(
            {
                "id": make_id("indian_scam_sms_synthetic", row["text"]),
                "text": row["text"],
                "language": language,
                "script": script,
                "label": label_map[row["label"]],
                "tactics": [],
                "source": "indian_scam_sms_synthetic",
                "source_id": str(row["row_id"]),
                "licence": "CC-BY-4.0",
                "provenance": "synthetic",
                "synthetic": True,
                "split": None,
            }
        )
    return _frame(rows)


def load_bengali() -> pd.DataFrame:
    label_map = {"smish": "scam", "normal": "legit", "promo": "legit"}
    rows = []
    for split in ("train", "validation", "test"):
        path = os.path.join(RAW_DIR, "bengali_sms_smishing", f"{split}-00000-of-00001.parquet")
        frame = pd.read_parquet(path)
        for index, row in frame.iterrows():
            language, script = BENGALI_VARIETY_MAP[row["source"]]
            rows.append(
                {
                    "id": make_id("bengali_sms_smishing", row["text"]),
                    "text": row["text"],
                    "language": language,
                    "script": script,
                    "label": label_map[row["label"]],
                    "tactics": [],
                    "source": "bengali_sms_smishing",
                    "source_id": f"{split}:{index}",
                    "licence": "MIT",
                    "provenance": "unverified",
                    "synthetic": False,  # collection method unstated, not eligible for test
                    "split": None,
                }
            )
    return _frame(rows)


def load_templates() -> pd.DataFrame:
    from scripts.dataset_generator import DatasetGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        output = os.path.join(tmpdir, "templates.csv")
        state = random.getstate()
        random.seed(42)
        try:
            DatasetGenerator(lang_packs_dir="data/language_packs", output_path=output).generate()
        finally:
            random.setstate(state)
        raw = pd.read_csv(output)
    rows = [
        {
            "id": make_id("template_pack", row["text"]),
            "text": row["text"],
            "language": row["language"],
            "script": row["script"],
            "label": "scam" if row["label"] == "scam" else "legit",
            "tactics": [t.strip() for t in str(row["tactics"]).split(",") if t.strip() in TACTICS],
            "source": "template_pack",
            "source_id": str(row["id"]),
            "licence": "MIT",
            "provenance": "template",
            "synthetic": True,
            "split": None,
        }
        for _, row in raw.iterrows()
    ]
    return _frame(rows)


def load_generated_synthetic() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(SYNTHETIC_DIR, "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                mappings = {
                    "english": ("english", "latin"),
                    "hindi": ("hindi", item.get("script", "native")),
                    "tamil": ("tamil", item.get("script", "native")),
                    "telugu": ("telugu", item.get("script", "native")),
                    "bengali": ("bengali", item.get("script", "native")),
                }
                language, script = mappings[item["language"]]
                rows.append(
                    {
                        "id": make_id("generated_synthetic", item["text"]),
                        "text": item["text"],
                        "language": language,
                        "script": script,
                        "label": item["label"],
                        "tactics": [t for t in item.get("tactics", []) if t in TACTICS],
                        "source": "generated_synthetic",
                        "source_id": item.get("id", ""),
                        "licence": "project",
                        "provenance": "synthetic",
                        "synthetic": True,
                        "split": None,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=SCHEMA_FIELDS)
    return _frame(rows)


def load_test_rows() -> pd.DataFrame:
    if not os.path.exists(TEST_PATH):
        return pd.DataFrame(columns=SCHEMA_FIELDS)
    records = [json.loads(line) for line in open(TEST_PATH, encoding="utf-8") if line.strip()]
    return pd.DataFrame(records)


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and label conflicts on normalised text."""
    frame = frame.copy()
    frame["_key"] = frame["text"].map(normalize_text)
    conflicts = frame.groupby("_key")["label"].nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    dropped_conflicts = int(frame["_key"].isin(conflict_keys).sum())
    frame = frame[~frame["_key"].isin(conflict_keys)]
    frame["_priority"] = frame["source"].map(SOURCE_PRIORITY)
    frame = frame.sort_values(["_priority"], kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("_key", keep="first")
    stats = {
        "dropped_label_conflicts": dropped_conflicts,
        "dropped_exact_duplicates": before - len(frame),
    }
    return frame.drop(columns=["_priority"]).reset_index(drop=True), stats


def near_duplicate_groups(texts: list[str], threshold: float = NEAR_DUP_THRESHOLD, neighbours: int = 6) -> np.ndarray:
    """Connected components of the "cosine similarity >= threshold" graph."""
    if len(texts) == 1:
        return np.zeros(1, dtype=int)
    vectorizer = TfidfVectorizer(**NEAR_DUP_VECTORIZER)
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return np.arange(len(texts))
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(neighbours, len(texts)))
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    parent = np.arange(len(texts))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in range(len(texts)):
        for distance, column in zip(distances[row], indices[row]):
            if column != row and 1.0 - distance >= threshold:
                union(row, int(column))
    roots = np.array([find(node) for node in range(len(texts))])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def assign_splits(frame: pd.DataFrame, groups: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    """Assign whole near-duplicate groups to train/val, stratified per cell."""
    splits = np.full(len(frame), "train", dtype=object)
    frame = frame.reset_index(drop=True)
    strata = frame["language"] + "|" + frame["script"] + "|" + frame["label"]
    group_frame = pd.DataFrame({"group": groups, "stratum": strata}).drop_duplicates()
    for stratum, members in group_frame.groupby("stratum"):
        member_groups = list(members["group"])
        sizes = {group: int((groups == group).sum()) for group in member_groups}
        target = val_fraction * sum(sizes.values())
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(member_groups)
        member_groups.sort(key=lambda group: -sizes[group])
        assigned = 0.0
        for group in member_groups:
            if assigned + sizes[group] / 2 <= target:
                mask = groups == group
                splits[mask] = "val"
                assigned += sizes[group]
    return splits


def write_jsonl(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in frame.to_dict("records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(val_fraction: float = 0.1, seed: int = 42, near_dup_threshold: float = NEAR_DUP_THRESHOLD):
    pool = pd.concat(
        [load_uci(), load_bengali(), load_ridham(), load_generated_synthetic(), load_templates()],
        ignore_index=True,
    )
    reserved = int((pool["split_hint"] == RESERVE_HINT).sum())
    pool = pool[pool["split_hint"] != RESERVE_HINT].drop(columns=["split_hint"]).reset_index(drop=True)
    pool, dedup_stats = deduplicate(pool)
    dedup_stats["held_out_for_test_reserve"] = reserved

    test = load_test_rows()
    combined = pd.concat([pool, test], ignore_index=True)
    groups = near_duplicate_groups(combined["text"].tolist(), threshold=near_dup_threshold)

    test_mask = np.zeros(len(combined), dtype=bool)
    test_mask[len(pool):] = True
    test_groups = set(groups[test_mask].tolist())
    excluded = np.isin(groups[: len(pool)], list(test_groups)) & ~test_mask[: len(pool)]
    pool_groups = groups[: len(pool)][~excluded]
    pool = pool[~excluded].reset_index(drop=True)

    splits = assign_splits(pool, pool_groups, val_fraction, seed)
    stats = {
        "rows": int(len(pool)),
        "train_rows": int((splits == "train").sum()),
        "val_rows": int((splits == "val").sum()),
        "near_duplicate_groups": int(len(set(groups.tolist()))),
        "test_excluded_rows": int(excluded.sum()),
        "near_dup_threshold": near_dup_threshold,
        **dedup_stats,
    }
    for column in ("language", "script", "label", "source", "provenance"):
        stats[f"by_{column}"] = pool[column].value_counts().to_dict()
    stats["by_split_cell"] = {
        "|".join(map(str, key)): int(value)
        for key, value in pool.assign(split=splits)
        .groupby(["split", "language", "script", "label"])
        .size()
        .items()
    }
    return pool, splits, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=float, default=NEAR_DUP_THRESHOLD)
    args = parser.parse_args()

    pool, splits, stats = build(args.val_fraction, args.seed, args.near_dup_threshold)
    train = pool[splits == "train"].drop(columns=["split"])
    val = pool[splits == "val"].drop(columns=["split"])
    write_jsonl(train, os.path.join(PROCESSED_DIR, "train.jsonl"))
    write_jsonl(val, os.path.join(PROCESSED_DIR, "val.jsonl"))
    with open(os.path.join(PROCESSED_DIR, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(stats, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
