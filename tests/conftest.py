import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.models.tfidf import TfidfScamModel


@pytest.fixture(scope="session")
def tiny_model() -> TfidfScamModel:
    """A baseline trained on a handful of rows; enough to exercise the interface."""
    rows = []
    # Enough urgency positives to clear the 10-positive floor for a tactic head.
    for index in range(12):
        rows.append({
            "text": f"Urgent {index}: your account will be blocked in {index + 1} minutes, click bit.ly/abc{index} now",
            "label": "scam",
            "tactics": ["urgency", "suspicious_link"],
            "language": "english",
            "script": "latin",
        })
    rows += [
        {"text": "Your SBI account was credited with Rs 5,000 via UPI. Ref 610928",
         "label": "legit", "tactics": [], "language": "english", "script": "latin"},
        {"text": "Your login OTP is 482913. Never share it with anyone.",
         "label": "legit", "tactics": [], "language": "english", "script": "latin"},
        {"text": "Delivery scheduled for tomorrow between 10am and 2pm",
         "label": "legit", "tactics": [], "language": "english", "script": "latin"},
    ]
    return TfidfScamModel(c_grid=[1.0]).fit(rows)
