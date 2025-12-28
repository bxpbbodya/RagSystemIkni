# core/security.py
from __future__ import annotations
import re

def mask_secret(value: str) -> str:
    """Return a masked version of a secret for display (never show full)."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]

def sanitize_for_logs(text: str) -> str:
    """Remove anything that looks like a password/token from logs."""
    if not text:
        return ""
    text = re.sub(r"password\s*=\s*\S+", "password=***", text, flags=re.IGNORECASE)
    text = re.sub(r"token\s*=\s*\S+", "token=***", text, flags=re.IGNORECASE)
    return text
