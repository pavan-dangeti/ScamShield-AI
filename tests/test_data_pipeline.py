"""Tests for the data pipeline: scrubbing, deduplication, splits, test-set freezing."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import freeze_test, make_test_candidates
from scripts.build_dataset import (
    NEAR_DUP_THRESHOLD,
    assign_splits,
    deduplicate,
    near_duplicate_groups,
    normalize_text,
)
from scripts.generate_synthetic import validate
from src.pii import scrub_text
from src.detector import ScriptLanguageDetector

# Two messages from the same template with different placeholders are exactly the
# leakage pattern that inflated the pre-upgrade metrics, so the fixtures use that shape.
SCAM_TEMPLATE = "Jaldi karein! Apna {bank} electricity bill {time} ke andar pay karein nahi to aaj raat light cut jayegi."
SCAM_VARIANTS = [
    SCAM_TEMPLATE.format(bank="SBI", time="10 minutes"),
    SCAM_TEMPLATE.format(bank="HDFC Bank", time="30 mins"),
]
UNRELATED = [
    "Aapke account XXXX4921 me Rs 5,000 credit kiye gaye hain. - State Bank of India",
    "FreeMsg: Txt BOB to 80086 for a free ringtone. Reply TONE to 80086",
]
FRAMEWORK_URLS = SCAM_VARIANTS + UNRELATED


def make_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- scrubbing -------------------------------------------------------------


def test_scrub_masks_phone_email_and_upi():
    scrubbed = scrub_text("Call +91 98765 43210 or mail bob.singh@example.com or pay raju@ybl")
    assert "[PHONE]" in scrubbed
    assert "[EMAIL]" in scrubbed
    assert "[UPI_ID]" in scrubbed
    assert "98765" not in scrubbed and "bob.singh" not in scrubbed and "raju" not in scrubbed


def test_scrub_keeps_scam_signals():
    text = "Urgent! Your account will be blocked in 10 minutes. Click http://bit.ly/abc123 now"
    scrubbed = scrub_text(text)
    assert "bit.ly/abc123" in scrubbed  # the link itself is the signal
    assert "blocked" in scrubbed and "Urgent" in scrubbed


def test_scrub_strips_link_query_but_keeps_path():
    assert scrub_text("go to https://sbi-kyc.example.net/verify?token=abc123xyz") == (
        "go to https://sbi-kyc.example.net/verify"
    )


def test_scrub_masks_long_identifiers_but_keeps_amounts():
    assert scrub_text("aadhaar 123456789012 and Rs 5,000 due") == "aadhaar [ID] and Rs 5,000 due"


# --- normalisation and dedup ----------------------------------------------


def test_normalize_collapses_whitespace_and_nfkc():
    assert normalize_text("  ur账户   10\tminutes\n") == normalize_text("ur账户 10 minutes")


def test_deduplicate_drops_exact_duplicates():
    rows = [
        {"text": "hello world", "label": "scam", "source": "generated_synthetic"},
        {"text": "hello  world", "label": "scam", "source": "template_pack"},
    ]
    frame, stats = deduplicate(make_frame(rows))
    assert len(frame) == 1
    assert stats["dropped_exact_duplicates"] == 1


def test_deduplicate_drops_conflicting_labels():
    rows = [
        {"text": "ambiguous message", "label": "scam", "source": "uci_sms_spam"},
        {"text": "ambiguous message", "label": "legit", "source": "template_pack"},
    ]
    frame, stats = deduplicate(make_frame(rows))
    assert frame.empty
    assert stats["dropped_label_conflicts"] == 2


# --- leakage ---------------------------------------------------------------


def test_near_duplicates_are_grouped():
    groups = near_duplicate_groups(FRAMEWORK_URLS)
    assert groups[0] == groups[1]  # same template, different placeholders
    assert groups[0] != groups[2]  # different family
    assert groups[0] != groups[3]


def test_split_keeps_near_duplicate_groups_together():
    frame = make_frame(
        [
            {"language": "english", "script": "latin", "label": label, "text": text}
            for text in FRAMEWORK_URLS
            for label in (["scam"] * 2 + ["legit"] * 2)
        ]
    )
    groups = near_duplicate_groups(frame["text"].tolist())
    splits = assign_splits(frame, groups, val_fraction=0.5, seed=1)
    for group in set(groups.tolist()):
        assert len(set(splits[groups == group].tolist())) == 1


def test_every_val_row_has_a_train_near_duplicate_below_threshold():
    """The leakage guard: no val message may closely match any train message."""
    frame = make_frame(
        [
            {"language": "english", "script": "latin", "label": "scam", "text": text}
            for text in FRAMEWORK_URLS
        ]
    )
    groups = near_duplicate_groups(frame["text"].tolist())
    splits = assign_splits(frame, groups, val_fraction=0.5, seed=1)
    train_groups = set(groups[splits == "train"].tolist())
    val_groups = set(groups[splits == "val"].tolist())
    assert not (train_groups & val_groups)


# --- synthetic generation quality gates ------------------------------------


def test_validate_rejects_wrong_script():
    seen: set[str] = set()
    payload = {"text": "आपका खाता 10 मिनट में ब्लॉक हो जाएगा कृपया तुरंत क्लिक करें", "tactics": ["urgency"]}
    assert validate(payload, "tamil", "native", "scam", seen) is None


def test_validate_rejects_foreign_script_contamination():
    seen: set[str] = set()
    payload = {"text": "आपका खाता ब्लॉक 立即点击这里 account blocked hurry", "tactics": ["urgency"]}
    assert validate(payload, "hindi", "native", "scam", seen) is None


def test_validate_rejects_legit_that_asks_for_credentials():
    seen: set[str] = set()
    payload = {"text": "Your account will be blocked. Please share your OTP to verify your account", "tactics": []}
    assert validate(payload, "english", "latin", "legit", seen) is None


def test_validate_accepts_hard_negative():
    seen: set[str] = set()
    payload = {"text": "Your login OTP is 482913. Never share it with anyone.", "tactics": []}
    assert validate(payload, "english", "latin", "legit", seen) is not None


def test_validate_rejects_placeholder_artifacts():
    seen: set[str] = set()
    payload = {"text": "Click here: [suspicious_link] your account blocked hurry now", "tactics": ["urgency"]}
    assert validate(payload, "english", "latin", "scam", seen) is None


# --- candidate pool and freezing -------------------------------------------


def test_candidate_pool_is_real_only(monkeypatch, tmp_path):
    uci = make_frame([
        {"id": "uci-a", "text": "Free prize click bit.ly/win now", "label": "scam", "source": "uci_sms_spam",
         "source_id": "1", "split_hint": "test_reserve"},
        {"id": "uci-b", "text": "See you at the office tomorrow", "label": "legit", "source": "uci_sms_spam",
         "source_id": "2", "split_hint": "test_reserve"},
        {"id": "uci-c", "text": "This row is training material", "label": "legit", "source": "uci_sms_spam",
         "source_id": "3", "split_hint": "pool"},
    ])
    monkeypatch.setattr(make_test_candidates, "load_uci", lambda: uci)
    monkeypatch.setattr(make_test_candidates, "used_texts", lambda: set())
    monkeypatch.setattr(make_test_candidates, "CANDIDATE_DIR", str(tmp_path))
    monkeypatch.setattr(make_test_candidates, "USER_IMPORT", str(tmp_path / "user_messages.jsonl"))
    monkeypatch.setattr(make_test_candidates, "CANDIDATES", str(tmp_path / "candidates.jsonl"))
    monkeypatch.setattr(
        make_test_candidates,
        "load_user_messages",
        lambda: [{"id": "user-1", "text": "Unga account block aagidum click pannunga", "language": "tamil", "script": "romanized"}],
    )
    monkeypatch.setattr(sys, "argv", ["make_test_candidates", "--per-class", "1"])
    make_test_candidates.main()
    rows = [json.loads(line) for line in open(tmp_path / "candidates.jsonl", encoding="utf-8")]
    assert rows and all(row["provenance"] == "real" and row["synthetic"] is False for row in rows)
    assert not any("training material" in row["text"] for row in rows)  # reserve only


def _write_pool(tmp_path, rows):
    pool = tmp_path / "candidates.jsonl"
    pool.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _run_freeze(tmp_path, monkeypatch, candidates, labels):
    _write_pool(tmp_path, candidates)
    label_file = tmp_path / "labels.jsonl"
    label_file.write_text("\n".join(json.dumps(row) for row in labels) + "\n", encoding="utf-8")
    monkeypatch.setattr(freeze_test, "CANDIDATES", str(tmp_path / "candidates.jsonl"))
    monkeypatch.setattr(freeze_test, "TEST_PATH", str(tmp_path / "test.jsonl"))
    monkeypatch.setattr(freeze_test, "TEST_SET_DOC", str(tmp_path / "test_set.md"))
    monkeypatch.setattr(freeze_test, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["freeze_test", "--labels", str(label_file), "--min-per-cell", "1"])
    freeze_test.main()


def test_freeze_drops_synthetic_rows(tmp_path, monkeypatch):
    candidate = {
        "id": "test-1", "text": "Win a free prize click now", "language": "english", "script": "latin",
        "label": None, "tactics": [], "source": "generated_synthetic", "provenance": "synthetic",
        "synthetic": True, "cell": "english|latin",
    }
    label = {"id": "test-1", "text": candidate["text"], "label": "scam", "tactics": ["false_reward"]}
    with pytest.raises(SystemExit):
        _run_freeze(tmp_path, monkeypatch, [candidate], [label])


def test_freeze_drops_scam_without_tactic(tmp_path, monkeypatch):
    candidate = {
        "id": "test-1", "text": "Free prize click now", "language": "english", "script": "latin",
        "label": None, "tactics": [], "source": "uci_sms_spam", "provenance": "real",
        "synthetic": False, "cell": "english|latin",
    }
    label = {"id": "test-1", "text": candidate["text"], "label": "scam", "tactics": []}
    with pytest.raises(SystemExit):
        _run_freeze(tmp_path, monkeypatch, [candidate], [label])


def test_freeze_keeps_valid_labels_and_writes_test_set(tmp_path, monkeypatch):
    candidates = [
        {"id": f"test-{i}", "text": text, "language": "english", "script": "latin", "label": None,
         "tactics": [], "source": "uci_sms_spam", "provenance": "real", "synthetic": False,
         "cell": "english|latin"}
        for i, text in enumerate(["Free prize click now", "Your account was credited with Rs 500"])
    ]
    labels = [
        {"id": "test-0", "text": candidates[0]["text"], "label": "scam", "tactics": ["false_reward"]},
        {"id": "test-1", "text": candidates[1]["text"], "label": "legit", "tactics": []},
    ]
    _run_freeze(tmp_path, monkeypatch, candidates, labels)
    frozen = [json.loads(line) for line in open(tmp_path / "test.jsonl", encoding="utf-8")]
    assert len(frozen) == 2
    assert all(row["split"] == "test" and row["synthetic"] is False for row in frozen)
    assert "Total rows: **2**" in (tmp_path / "test_set.md").read_text(encoding="utf-8")


def test_freeze_reports_self_agreement_and_uci_disagreements(tmp_path, monkeypatch):
    texts = ["Free prize click now", "Your account was credited with Rs 500", "50% off pizza this weekend"]
    candidates = [
        {"id": f"test-uci-{i}", "text": text, "language": "english", "script": "latin", "label": None,
         "tactics": [], "source": "uci_sms_spam", "provenance": "real", "synthetic": False,
         "cell": "english|latin"}
        for i, text in enumerate(texts)
    ]
    labels = [
        {"id": "test-uci-0", "text": texts[0], "label": "scam", "tactics": ["false_reward"], "pass": "primary",
         "confidence": "sure", "annotator": "ab", "seconds": 3.0},
        {"id": "test-uci-1", "text": texts[1], "label": "legit", "tactics": [], "pass": "primary",
         "confidence": "unsure", "annotator": "ab", "seconds": 5.0},
        {"id": "test-uci-2", "text": texts[2], "label": "legit", "tactics": [], "pass": "primary",
         "confidence": "sure", "annotator": "ab", "seconds": 4.0},
        {"id": "test-uci-0", "text": texts[0], "label": "scam", "tactics": ["false_reward"], "pass": "recheck"},
        {"id": "test-uci-1", "text": texts[1], "label": "legit", "tactics": [], "pass": "recheck"},
    ]
    # UCI calls the pizza offer spam; the guidelines call marketing legitimate.
    uci = pd.DataFrame({"id": ["uci-0", "uci-1", "uci-2"], "label": ["scam", "legit", "scam"]})
    monkeypatch.setattr(freeze_test, "load_uci", lambda: uci)
    _run_freeze(tmp_path, monkeypatch, candidates, labels)

    frozen = [json.loads(line) for line in open(tmp_path / "test.jsonl", encoding="utf-8")]
    assert len(frozen) == 3, "re-check rows must not become extra test rows"
    assert {row["id"]: row["confidence"] for row in frozen}["test-uci-1"] == "unsure"
    doc = (tmp_path / "test_set.md").read_text(encoding="utf-8")
    assert "Re-checked items: 2; same decision: 2 (100.0%)" in doc
    assert "Cohen's kappa, scam/legit: 1.000" in doc
    assert "Compared rows: 3; disagreements: 1" in doc
    assert "| test-uci-2 | legit | scam | 50% off pizza this weekend |" in doc


# --- language packs (regression for the path bug) -------------------------


def test_detector_reads_language_packs_from_data_directory():
    detector = ScriptLanguageDetector("data/language_packs")
    tanglish = detector.detect("Unga account block aagidum, ippo click pannunga")
    assert tanglish["script_type"] == "romanized"
    assert tanglish["language_guess"] == "Tamil"
    devanagari = detector.detect("जल्दी करें! आपका खाता ब्लॉक हो जाएगा")
    assert devanagari["script_type"] == "native"
    assert devanagari["language_guess"] == "Hindi"
