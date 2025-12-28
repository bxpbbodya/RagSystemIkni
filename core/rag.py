# core/rag.py
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict

from .sources import SourceChunk
from .llm import LLMSettings, chat_completion


def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def _snip(text: str, max_len: int = 320) -> str:
    text = _clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def _extract_short_summary(text: str, max_sentences: int = 2) -> str:
    t = _clean_text(text)
    if not t:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    if not parts:
        return _snip(t, 220)
    return " ".join(parts[:max_sentences])


def _format_sources_md(retrieved: List[Tuple[SourceChunk, float]], top: int = 5) -> str:
    lines = []
    for i, (chunk, score) in enumerate(retrieved[:top], start=1):
        url = chunk.url or ""
        title = chunk.title or "джерело"
        src_type = chunk.source_type
        lines.append(f"{i}. [{title}]({url}) — `{src_type}` • score={score:.3f}")
    return "\n".join(lines)


def _build_context(retrieved: List[Tuple[SourceChunk, float]], max_chars: int = 6000) -> str:
    """
    Build compact RAG context from top chunks.
    """
    ctx_parts = []
    total = 0
    for i, (chunk, score) in enumerate(retrieved[:6], start=1):
        piece = f"[{i}] ({chunk.source_type}) {chunk.title}\nURL: {chunk.url}\nTEXT: {chunk.text}\n"
        piece = piece.strip() + "\n\n"
        if total + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        total += len(piece)
    return "".join(ctx_parts)


def make_answer_no_llm(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> str:
    """
    Better offline MVP answer:
    - short summary from best chunk
    - evidence list
    """
    if not retrieved:
        return (
            f"**Запит:** {query}\n\n"
            "❌ **Нічого не знайдено у локальній базі.**\n\n"
            "Спробуй інший запит або натисни **Sync knowledge base**, щоб оновити дані."
        )

    best_text = retrieved[0][0].text

    lines = []
    lines.append(f"**Запит:** {query}\n")
    lines.append("### ✅ Відповідь (offline / без LLM)")
    lines.append(_extract_short_summary(best_text, max_sentences=2))
    lines.append("\n---\n")
    lines.append("### 📌 Джерела")
    lines.append(_format_sources_md(retrieved, top=5))
    return "\n".join(lines)


def make_answer_with_llm(
    query: str,
    retrieved: List[Tuple[SourceChunk, float]],
    llm: LLMSettings,
) -> str:
    """
    RAG answer with LLM:
    - use retrieved chunks as context
    - ask model to answer only from provided sources
    - include sources block
    """
    if not retrieved:
        return make_answer_no_llm(query, retrieved)

    context = _build_context(retrieved)

    system = (
        "Ти — помічник для студентів ІКНІ ЛПНУ. "
        "Відповідай українською. "
        "Використовуй ТІЛЬКИ інформацію з наданого контексту. "
        "Якщо у контексті немає відповіді — чесно скажи, що даних недостатньо. "
        "Відповідь роби коротко і структуровано (1–6 пунктів), без води."
    )

    user = (
        f"Питання: {query}\n\n"
        f"Контекст (джерела):\n{context}\n\n"
        "Згенеруй відповідь. Наприкінці додай короткий блок 'Джерела:' "
        "і перерахуй номери [1], [2]... які ти реально використав."
    )

    try:
        content = chat_completion(
            llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        # If LLM failed, fallback to offline
        return (
            f"⚠️ **LLM помилка:** `{e}`\n\n"
            + make_answer_no_llm(query, retrieved)
        )

    # Add real clickable sources (from retrieval)
    sources_md = _format_sources_md(retrieved, top=5)

    return (
        f"**Запит:** {query}\n\n"
        f"### ✅ Відповідь (LLM)\n"
        f"{content.strip()}\n\n"
        f"---\n"
        f"### 📌 Джерела (retrieval топ-5)\n"
        f"{sources_md}"
    )
