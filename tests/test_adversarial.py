"""Tests for the adversarial perturbation suite.

The perturbation families are used to generate both the robustness evaluation and
the augmented training set, so a silent no-op or a label-changing bug would quietly
invalidate Phase 4. These tests pin both properties.
"""

from __future__ import annotations

import pytest

from scripts.adversarial import FAMILIES, build_adversarial_split, perturb

SCAM = "Urgent! Your SBI account will be blocked in 10 minutes, click http://bit.ly/kyc123 to verify."
LEGIT = "Your account was credited with Rs 5,000 via UPI. Ref 610928"


@pytest.mark.parametrize("family", FAMILIES)
def test_perturbation_changes_ascii_text(family):
    assert perturb(SCAM, family, 42) != SCAM, f"{family} did not change the message"


@pytest.mark.parametrize("family", FAMILIES)
def test_perturbation_is_deterministic(family):
    assert perturb(SCAM, family, 42) == perturb(SCAM, family, 42)


@pytest.mark.parametrize("family", FAMILIES)
def test_perturbation_varies_with_seed(family):
    variants = {perturb(SCAM, family, seed) for seed in (1, 2, 3, 4, 5)}
    assert len(variants) > 1, f"{family} ignored the seed"


@pytest.mark.parametrize("family", FAMILIES)
def test_perturbation_preserves_the_label_fields(family):
    rows = [
        {"id": "a", "text": SCAM, "label": "scam", "tactics": ["urgency"], "language": "english", "script": "latin"},
        {"id": "b", "text": LEGIT, "label": "legit", "tactics": [], "language": "english", "script": "latin"},
    ]
    perturbed, unchanged = build_adversarial_split(rows, family)
    assert len(perturbed) == len(rows)
    assert [r["label"] for r in perturbed] == ["scam", "legit"]
    assert perturbed[0]["tactics"] == ["urgency"]
    assert all(r["attack"] == family for r in perturbed)
    assert 0 <= unchanged <= len(rows)


def test_disguised_link_never_invents_a_link():
    """Appending a phishing link to a legitimate notification would change its label."""
    result = perturb(LEGIT, "disguised_link", 42)
    assert "http" not in result.lower()


def test_disguised_link_obfuscates_an_existing_link():
    result = perturb(SCAM, "disguised_link", 42)
    assert "bit.ly" not in result or "b" not in result.split("bit.ly")[0][-3:]


def test_spacing_inserts_a_real_zero_width_character():
    result = perturb(SCAM, "spacing", 42)
    assert "" in result or "  " in result


def test_native_script_rows_are_reported_as_unchanged_not_silently_kept():
    devanagari = "आपका खाता 10 मिनट में ब्लॉक हो जाएगा, तुरंत क्लिक करें"
    perturbed, unchanged = build_adversarial_split(
        [{"id": "x", "text": devanagari, "label": "scam", "tactics": ["urgency"],
          "language": "hindi", "script": "native"}],
        "char_swap",
    )
    assert perturbed[0]["text"] == devanagari
    assert unchanged == 1


def test_augmented_rows_are_traceable_to_their_source():
    rows = [
        {"id": f"src-{index}", "text": SCAM, "label": "scam", "tactics": ["urgency"],
         "language": "english", "script": "latin"}
        for index in range(5)
    ]
    perturbed, _ = build_adversarial_split(rows, "char_swap", 42)
    assert len(perturbed) == 5
    assert all(row["attack"] == "char_swap" for row in perturbed)
    assert all(row["id"].startswith("src-") for row in perturbed)
    assert all(row["label"] == "scam" for row in perturbed)
