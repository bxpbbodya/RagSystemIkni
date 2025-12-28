# core/sources.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    title: str
    source_type: str   # "local" | "lpnu" | "tg" | "vns"
    url: Optional[str] = None
    date: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
