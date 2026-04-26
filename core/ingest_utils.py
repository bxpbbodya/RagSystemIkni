from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .sources import SourceChunk


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def now_iso_date() -> str:
    return datetime.now().date().isoformat()


def make_doc_id(source_type: str, url: str, title: str, raw_text: str) -> str:
    clean_text = normalize_whitespace(raw_text)
    base = f"{source_type}|{url}|{title}|{sha1(clean_text)[:16]}"
    return sha1(base)[:20]


def make_chunk_id(source_type: str, doc_id: str, part_index: int) -> str:
    return f"{source_type}_{doc_id}_{part_index:03d}"


def split_paragraphs(text: str, min_len: int = 50) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in parts if p.strip() and len(p.strip()) >= min_len]
    if paragraphs:
        return paragraphs

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    merged: List[str] = []
    buffer = ""
    for line in lines:
        if len(buffer) + len(line) + 1 <= 450:
            buffer += (" " if buffer else "") + line
        else:
            if len(buffer.strip()) >= min_len:
                merged.append(buffer.strip())
            buffer = line
    if len(buffer.strip()) >= min_len:
        merged.append(buffer.strip())
    return merged


def chunk_text(
        text: str,
        chunk_size: int = 600,
        overlap: int = 120,
) -> List[str]:
    """
    Розбиває текст на шматки, використовуючи ієрархію сепараторів.
    Оптимізовано для пам'яті: уникаємо рекурсії та зайвого копіювання.
    """
    text = normalize_whitespace(text)
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    # Сепаратори від найбільш пріоритетних (абзаци) до найменших (пробіли)
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    chunks = []

    start_ptr = 0
    text_len = len(text)

    while start_ptr < text_len:
        # Визначаємо кінець поточного вікна
        end_ptr = start_ptr + chunk_size

        if end_ptr >= text_len:
            chunks.append(text[start_ptr:].strip())
            break

        # Шукаємо найкраще місце для розриву всередині вікна
        split_idx = -1
        for sep in separators:
            # Шукаємо сепаратор з кінця вікна назад до початку
            found_idx = text.rfind(sep, start_ptr, end_ptr)
            if found_idx != -1:
                # Знайшли найкращий розрив
                split_idx = found_idx + len(sep)
                break

        # Якщо сепараторів не знайдено (дуже довге слово), ріжемо по ліміту
        if split_idx <= start_ptr:
            split_idx = end_ptr

        chunk = text[start_ptr:split_idx].strip()
        if chunk:
            chunks.append(chunk)

        # Зміщуємо вказівник назад на величину оверлапу
        start_ptr = max(start_ptr + 1, split_idx - overlap)

    return chunks

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    def split_long_paragraph(paragraph: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        local_chunks: List[str] = []
        local_buffer = ""
        for sentence in sentences:
            if len(local_buffer) + len(sentence) + 1 <= chunk_size:
                local_buffer += (" " if local_buffer else "") + sentence
            else:
                if local_buffer.strip():
                    local_chunks.append(local_buffer.strip())
                local_buffer = sentence
        if local_buffer.strip():
            local_chunks.append(local_buffer.strip())
        return local_chunks or [paragraph.strip()]

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            flush_buffer()
            chunks.extend(split_long_paragraph(paragraph))
            continue

        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            flush_buffer()
            buffer = paragraph

    flush_buffer()

    if overlap > 0 and chunks:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())
        chunks = overlapped

    return chunks


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
    debug: bool = False,
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
    total_parts = len(parts)
    for i, part in enumerate(parts):
        chunk_id = make_chunk_id(source_type, doc_id, i)
        chunk_extra = dict(extra_dict)
        chunk_extra["chunk_index"] = i
        chunk_extra["chunk_count"] = total_parts
        chunk_extra["word_count"] = len(part.split())
        chunks.append(
            SourceChunk(
                chunk_id=chunk_id,
                text=part,
                title=title,
                source_type=source_type,
                url=url,
                date=date or now_iso_date(),
                extra=chunk_extra,
            )
        )

    if debug:
        print(f"[DEBUG] {url} -> {len(chunks)} chunks")

    return chunks
