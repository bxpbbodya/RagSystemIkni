# core/sources.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    title: str
    source_type: str   # "local" | "lpnu" | "tg" | "vns"
    url: Optional[str] = None
    date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # ✅ always dict

    def __post_init__(self):
        # In case someone passes extra=None or some junk
        if self.extra is None:
            self.extra = {}
