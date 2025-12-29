# core/ingest_utils.py
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Set, Tuple

from .sources import SourceChunk


# -----------------------------
# Text helpers
# -----------------------------
def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def now_iso_date() -> str:
    return datetime.now().date().isoformat()


# -----------------------------
# Stable doc_id / chunk_id helpers
# -----------------------------
def make_doc_id(source_type: str, url: str, title: str, raw_text: str) -> str:
    """
    Stable doc_id:
      - based on source_type + url + title + small hash of content
    """
    base = f"{source_type}|{url}|{title}|{sha1(raw_text)[:10]}"
    return sha1(base)[:16]


def make_chunk_id(source_type: str, doc_id: str, part_index: int) -> str:
    return f"{source_type}_{doc_id}_{part_index:03d}"


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Simple character chunking with overlap.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


# -----------------------------
# JSONL helpers (safe)
# -----------------------------
def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON safely. Returns None on failure.
    """
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def load_jsonl(
    path: Path,
    *,
    limit: Optional[int] = None,
    ignore_errors: bool = True,
) -> List[Dict[str, Any]]:
    """
    Safe JSONL loader.
    - skips broken lines
    - returns last `limit` items if limit provided
    """
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue

            obj = safe_json_loads(line)
            if obj is None:
                if ignore_errors:
                    continue
                raise ValueError("Broken JSONL line encountered")

            rows.append(obj)

    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def append_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    """
    Append items to JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            try:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
            except Exception:
                # if item isn't json-serializable, skip
                continue


# -----------------------------
# Dedup helpers
# -----------------------------
def existing_chunk_ids(cache_path: Path) -> Set[str]:
    """
    Read cache JSONL and return set of chunk_id.
    - safe parsing
    - skips broken lines
    - never includes None
    """
    ids: Set[str] = set()
    if not cache_path.exists():
        return ids

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = safe_json_loads(line)
            if not obj:
                continue
            cid = obj.get("chunk_id")
            if cid:
                ids.add(str(cid))
    return ids


def existing_doc_ids(
    cache_path: Path,
    *,
    extra_key: str = "doc_id",
) -> Set[str]:
    """
    Scan cache JSONL and return set of doc_id stored in extra[doc_id].
    Works even if extra is None.
    """
    ids: Set[str] = set()
    if not cache_path.exists():
        return ids

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = safe_json_loads(line)
            if not obj:
                continue
            extra = obj.get("extra") or {}
            did = extra.get(extra_key)
            if did:
                ids.add(str(did))
    return ids


def existing_extra_keys(
    cache_path: Path,
    *,
    key: str,
) -> Set[str]:
    """
    Generic helper: returns set of extra[key] from cache.
    Example: key="message_key" for Telegram dedup.
    """
    values: Set[str] = set()
    if not cache_path.exists():
        return values

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = safe_json_loads(line)
            if not obj:
                continue
            extra = obj.get("extra") or {}
            v = extra.get(key)
            if v:
                values.add(str(v))
    return values


# -----------------------------
# Chunk builder
# -----------------------------
def make_chunks_from_doc(
    *,
    source_type: str,
    url: str,
    title: str,
    raw_text: str,
    date: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
    chunk_size: int = 900,
    overlap: int = 120,
    doc_id: Optional[str] = None,
) -> List[SourceChunk]:
    """
    Convert a document into SourceChunk list with stable chunk_id.

    ✅ Fixes:
    - extra can be None
    - doc_id can be passed explicitly (for stable dedup)
    - extra is merged with doc_id + origin meta
    """
    raw_text = normalize_whitespace(raw_text)
    if not raw_text:
        return []

    url = (url or "").strip()
    title = (title or url or "document").strip()

    # stable doc_id (allow caller to provide explicit)
    if doc_id is None or not str(doc_id).strip():
        doc_id = make_doc_id(source_type, url, title, raw_text)
    doc_id = str(doc_id)

    parts = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    if not parts:
        return []

    # always dict
    extra_dict: Dict[str, Any] = dict(extra or {})
    extra_dict.setdefault("doc_id", doc_id)
    extra_dict.setdefault("source_type", source_type)

    chunks: List[SourceChunk] = []
    for i, part in enumerate(parts):
        chunk_id = make_chunk_id(source_type, doc_id, i)
        chunks.append(
            SourceChunk(
                chunk_id=chunk_id,
                text=part,
                title=title,
                source_type=source_type,
                url=url,
                date=date,
                extra=extra_dict,
            )
        )
    return chunks
