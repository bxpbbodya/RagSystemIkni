# core/sources.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import hashlib

@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    title: str
    source_type: str   # "local" | "lpnu" | "tg" | "vns" | "resource"
    url: Optional[str] = None
    date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # ✅ always dict

    def __post_init__(self):
        # Ensure extra is always a dict
        if self.extra is None:
            self.extra = {}

        # Generate a unique chunk_id if empty
        if not self.chunk_id:
            # Hash title + text for deterministic ID
            base = (self.title or "") + (self.text or "")
            self.chunk_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]