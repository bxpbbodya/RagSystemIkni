# core/ingest_utils.py
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from .sources import SourceChunk


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_doc_id(source_type: str, url: str, title: str, raw_text: str) -> str:
    base = f"{source_type}|{url}|{title}|{sha1(raw_text)[:10]}"
    return sha1(base)[:16]


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for MVP; can be upgraded to token-based later.
    """
    text = normalize_whitespace(text)
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def append_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def existing_chunk_ids(cache_path: Path) -> set[str]:
    ids = set()
    if not cache_path.exists():
        return ids
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj.get("chunk_id"))
            except Exception:
                continue
    return ids


def make_chunks_from_doc(
    *,
    source_type: str,
    url: str,
    title: str,
    raw_text: str,
    date: Optional[str],
    extra: Optional[Dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[SourceChunk]:
    raw_text = normalize_whitespace(raw_text)
    if not raw_text:
        return []

    doc_id = make_doc_id(source_type, url, title, raw_text)
    parts = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

    chunks: List[SourceChunk] = []
    for i, part in enumerate(parts):
        chunk_id = f"{source_type}_{doc_id}_{i:03d}"
        chunks.append(SourceChunk(
            chunk_id=chunk_id,
            text=part,
            title=title or url,
            source_type=source_type,
            url=url,
            date=date,
            extra=extra or {}
        ))
    return chunks


def now_iso_date() -> str:
    return datetime.now().date().isoformat()
