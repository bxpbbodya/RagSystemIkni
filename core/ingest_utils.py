# core/ingest_utils.py (optimized)
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Set

from .sources import SourceChunk

# -----------------------------
# Text helpers
# -----------------------------
def normalize_whitespace(text: str) -> str:
    """Вирівнювання пробілів та символів нового рядка"""
    text = (text or "").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def now_iso_date() -> str:
    return datetime.now().date().isoformat()

# -----------------------------
# Stable doc_id / chunk_id helpers
# -----------------------------
def make_doc_id(source_type: str, url: str, title: str, raw_text: str) -> str:
    """Стабільний doc_id на основі source + url + title + частини sha1 тексту"""
    clean_text = normalize_whitespace(raw_text)
    base = f"{source_type}|{url}|{title}|{sha1(clean_text)[:16]}"
    return sha1(base)[:20]

def make_chunk_id(source_type: str, doc_id: str, part_index: int) -> str:
    return f"{source_type}_{doc_id}_{part_index:03d}"

# -----------------------------
# Better chunking
# -----------------------------
def split_paragraphs(text: str, min_len: int = 50) -> List[str]:
    """Розбивка на параграфи + відсікання коротких"""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= min_len]

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_para_len: int = 50
) -> List[str]:
    """Гібридний чанкінг: параграфи + sliding window + sentence-aware"""
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = split_paragraphs(text, min_len=min_para_len)
    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue

        # Спроба ділити на речення, щоб уникати обрізання посередині
        sentences = re.split(r'(?<=[.!?])\s+', para)
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) + 1 <= chunk_size:
                buffer += (" " if buffer else "") + sent
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = sent
        if buffer:
            chunks.append(buffer.strip())

        # Додаємо overlap між останніми chunk
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                overlap_text = chunks[i-1][-overlap:] + " " + chunks[i][:overlap]
                chunks[i] = overlap_text

    return chunks

# -----------------------------
# JSONL helpers
# -----------------------------
def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def load_jsonl(path: Path, *, limit: Optional[int] = None, ignore_errors: bool = True) -> List[Dict[str, Any]]:
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
                raise ValueError(f"Broken JSONL line in {path}")
            rows.append(obj)

    if limit and limit > 0:
        return rows[-limit:]
    return rows

def append_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            try:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception:
                continue

# -----------------------------
# Dedup helpers
# -----------------------------
def existing_chunk_ids(cache_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not cache_path.exists():
        return ids
    for obj in load_jsonl(cache_path):
        cid = obj.get("chunk_id")
        if cid:
            ids.add(str(cid).strip())
    return ids

def existing_doc_ids(cache_path: Path, *, extra_key: str = "doc_id") -> Set[str]:
    ids: Set[str] = set()
    if not cache_path.exists():
        return ids
    for obj in load_jsonl(cache_path):
        extra = obj.get("extra") or {}
        did = extra.get(extra_key)
        if did:
            ids.add(str(did).strip())
    return ids

# -----------------------------
# Chunk builder
# -----------------------------
def make_chunks_from_doc(
    *,
    source_type: str,
    url: str,
    title: str,
    raw_text: str,
    date: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    overlap: int = 100,
    doc_id: Optional[str] = None,
    debug: bool = False
) -> List[SourceChunk]:

    raw_text = normalize_whitespace(raw_text)
    if not raw_text:
        return []

    url = (url or "").strip()
    title = (title or url or "document").strip()

    if not doc_id:
        doc_id = make_doc_id(source_type, url, title, raw_text)
    doc_id = str(doc_id)

    parts = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    if not parts:
        return []

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
                date=date or now_iso_date(),
                extra=extra_dict,
            )
        )

    if debug:
        print(f"[DEBUG] {url} -> {len(chunks)} chunks")

    return chunks