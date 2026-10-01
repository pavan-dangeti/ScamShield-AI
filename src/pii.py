"""PII scrubbing for ScamShield messages.

Real messages can contain personal data (phone numbers, email/UPI handles,
account identifiers, tracking parameters in links).  Scam *patterns* must
survive scrubbing, so replacements are slot tokens such as ``[PHONE]`` that
preserve the shape of the message.
"""

from __future__ import annotations

import re

# Order matters: UPI handles are "word@word" and overlap with email.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_UPI = re.compile(r"\b[\w.-]{2,}@[A-Za-z][\w.-]{1,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_LONG_ID = re.compile(r"(?<!\d)\d{12,16}(?!\d)")
_URL = re.compile(r"(?:(?:https?://)|(?:www\.))[^\s]+|\b[\w-]+(?:\.[\w-]+)+(?:/[^\s]*)?")
_WHITESPACE = re.compile(r"[ \t\u00a0]+")


def scrub_url(url: str) -> str:
    """Keep scheme/host/path, drop query strings and fragments (may carry tokens)."""
    for separator in ("?", "#"):
        if separator in url:
            url = url.split(separator, 1)[0]
    return url


def scrub_text(text: str) -> str:
    """Replace personal identifiers with slot tokens. Returns the scrubbed text."""
    if not isinstance(text, str):
        return ""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _UPI.sub("[UPI_ID]", text)
    text = _LONG_ID.sub("[ID]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _URL.sub(lambda match: scrub_url(match.group(0)), text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()
