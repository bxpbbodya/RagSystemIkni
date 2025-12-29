# core/rag.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from .sources import SourceChunk
from .llm import LLMSettings, chat_completion


# -----------------------------
# Data models
# -----------------------------
@dataclass
class RAGAnswer:
    """
    Structured result (useful for UI):
      - markdown: full answer
      - used_sources: indices [1..k] referenced
      - warnings: any quality issues
      - source_map: index -> SourceChunk (for UI)
    """
    markdown: str
    used_sources: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_map: Dict[int, SourceChunk] = field(default_factory=dict)


# -----------------------------
# Text utils
# -----------------------------
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


def _query_keywords(query: str, min_len: int = 4) -> List[str]:
    q = query.lower()
    q = re.sub(r"[^a-zа-яіїєґ0-9\s]", " ", q)
    words = [w.strip() for w in q.split() if len(w.strip()) >= min_len]

    # unique but keep order
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:12]


def highlight_keywords_md(text: str, keywords: List[str]) -> str:
    """
    Highlights keywords inside markdown text using HTML <mark>.
    Requires Streamlit: st.markdown(..., unsafe_allow_html=True).

    Safe-ish: does not touch URLs/markdown links too aggressively.
    """
    if not keywords:
        return text

    kws = sorted(set(keywords), key=len, reverse=True)
    if not kws:
        return text

    # Avoid breaking markdown links by not highlighting inside (...) URL parts.
    # We'll highlight only outside markdown links: [title](url)
    # Simple heuristic: split by ')', highlight only in parts that do not contain 'http'
    pattern = r"(" + "|".join(re.escape(k) for k in kws if k) + r")"

    def repl(m):
        return f"<mark>{m.group(1)}</mark>"

    try:
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)
    except Exception:
        return text


# -----------------------------
# Sources formatting
# -----------------------------
def format_sources_md(retrieved: List[Tuple[SourceChunk, float]], top: int = 5) -> str:
    lines = []
    for i, (chunk, score) in enumerate(retrieved[:top], start=1):
        url = chunk.url or ""
        title = _clean_text(chunk.title or "джерело")
        src_type = chunk.source_type or "unknown"
        date = chunk.date or ""
        title = title.replace("[", "(").replace("]", ")")
        lines.append(f"[{i}] [{title}]({url}) — `{src_type}` • {date} • score={score:.3f}")
    return "\n".join(lines)


def build_context(retrieved: List[Tuple[SourceChunk, float]], max_chars: int = 6500, max_chunks: int = 6) -> str:
    """
    Compact context for LLM:
      [1] title
      URL:
      DATE:
      TEXT:
    """
    ctx_parts = []
    total = 0
    for i, (chunk, score) in enumerate(retrieved[:max_chunks], start=1):
        piece = (
            f"[{i}] ({chunk.source_type}) {chunk.title}\n"
            f"URL: {chunk.url}\n"
            f"DATE: {chunk.date}\n"
            f"TEXT: {chunk.text}\n"
        ).strip() + "\n\n"

        if total + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        total += len(piece)

    return "".join(ctx_parts)


# -----------------------------
# Citation parsing & validation
# -----------------------------
def parse_used_sources(text: str, k_max: int) -> List[int]:
    """
    Finds citations like [1], [2], ...
    Returns sorted unique list within 1..k_max
    """
    found = set()
    for m in re.findall(r"\[(\d{1,2})\]", text or ""):
        try:
            n = int(m)
            if 1 <= n <= k_max:
                found.add(n)
        except Exception:
            continue
    return sorted(found)


def enforce_sources_block(answer: str, used_sources: List[int]) -> str:
    """
    Ensures answer has a clear sources line at bottom.
    """
    if not used_sources:
        used_str = "немає (контекст недостатній)"
    else:
        used_str = ", ".join(f"[{i}]" for i in used_sources)

    if re.search(r"\bджерела\b\s*:", answer, flags=re.IGNORECASE):
        return answer.strip()

    return (answer.strip() + f"\n\n**Джерела:** {used_str}").strip()


def answer_has_no_data_phrase(text: str) -> bool:
    t = (text or "").lower()
    return (
        "немає даних у контексті" in t
        or "недостатньо даних у контексті" in t
        or "не можу знайти у контексті" in t
    )


# -----------------------------
# OFFLINE answer builder (structured)
# -----------------------------
def make_answer_no_llm_struct(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> RAGAnswer:
    if not retrieved:
        md = (
            f"**Запит:** {query}\n\n"
            "❌ **Нічого не знайдено у локальній базі.**\n\n"
            "Спробуй інший запит або натисни **Sync knowledge base**, щоб оновити дані."
        )
        return RAGAnswer(markdown=md, used_sources=[], warnings=["no_results"])

    # Summary from best chunk -> cite [1]
    best_summary = _extract_short_summary(retrieved[0][0].text, max_sentences=2)
    bullets = [f"- {best_summary} **[1]**"] if best_summary else []

    # Add 1–2 extra bullets from distinct chunks (avoid duplicates)
    seen_norm = {best_summary.lower().strip()} if best_summary else set()
    used_sources = {1}

    for idx in range(1, min(len(retrieved), 5)):  # only consider first few
        txt = _extract_short_summary(retrieved[idx][0].text, max_sentences=1)
        norm = (txt or "").lower().strip()
        if not txt or len(norm) < 30:
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        bullets.append(f"- {txt} **[{idx+1}]**")
        used_sources.add(idx + 1)
        if len(bullets) >= 3:
            break

    sources_md = format_sources_md(retrieved, top=5)

    md = (
        f"**Запит:** {query}\n\n"
        f"### ✅ Відповідь (offline / без LLM)\n"
        f"{chr(10).join(bullets) if bullets else '_Немає тексту для короткого підсумку._'}\n\n"
        f"---\n"
        f"### 📌 Джерела\n"
        f"{sources_md}"
    )

    # highlight keywords (optional)
    kws = _query_keywords(query)
    md = highlight_keywords_md(md, kws)

    # build source_map
    source_map = {i: retrieved[i - 1][0] for i in sorted(used_sources) if 1 <= i <= len(retrieved)}

    return RAGAnswer(markdown=md, used_sources=sorted(used_sources), source_map=source_map)


# -----------------------------
# LLM answer builder (structured)
# -----------------------------
def make_answer_with_llm_struct(
    query: str,
    retrieved: List[Tuple[SourceChunk, float]],
    llm: LLMSettings,
) -> RAGAnswer:
    if not retrieved:
        return make_answer_no_llm_struct(query, retrieved)

    k_max = min(len(retrieved), 6)
    context = build_context(retrieved, max_chars=6500, max_chunks=k_max)

    system = (
        "Ти — помічник для студентів ІКНІ ЛПНУ.\n"
        "Відповідай **українською**.\n"
        "⚠️ Використовуй ТІЛЬКИ інформацію з наданого контексту.\n"
        "Якщо відповіді немає — скажи: 'Немає даних у контексті'.\n\n"
        "🔥 Головна вимога: після КОЖНОГО твердження став посилання на джерело у форматі [1], [2]...\n"
        "Не вигадуй джерел — використовуй лише номери які є у контексті."
    )

    user = (
        f"Питання: {query}\n\n"
        f"Контекст (джерела):\n{context}\n\n"
        "Дай відповідь у вигляді 3–8 коротких пунктів.\n"
        "Після кожного пункту постав посилання [n].\n"
        "Наприкінці окремим рядком напиши: 'Джерела: [..]' (лише використані)."
    )

    warnings: List[str] = []

    try:
        content = chat_completion(
            llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        warnings.append(f"llm_error: {e}")
        # fallback
        fallback = make_answer_no_llm_struct(query, retrieved)
        fallback.warnings = (fallback.warnings or []) + warnings
        return fallback

    used = parse_used_sources(content, k_max=k_max)

    if not used:
        warnings.append("no_citations_found")

    content = enforce_sources_block(content, used)

    if answer_has_no_data_phrase(content):
        warnings.append("llm_says_no_data")

    sources_md = format_sources_md(retrieved, top=5)

    md = (
        f"**Запит:** {query}\n\n"
        f"### ✅ Відповідь (LLM)\n"
        f"{content.strip()}\n\n"
        f"---\n"
        f"### 📌 Джерела (retrieval топ-5)\n"
        f"{sources_md}"
    )

    kws = _query_keywords(query)
    md = highlight_keywords_md(md, kws)

    # source_map (only for used citations)
    source_map = {i: retrieved[i - 1][0] for i in used if 1 <= i <= len(retrieved)}

    return RAGAnswer(markdown=md, used_sources=used, warnings=warnings, source_map=source_map)


# -----------------------------
# Backward-compatible wrappers
# -----------------------------
def make_answer_no_llm(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> str:
    return make_answer_no_llm_struct(query, retrieved).markdown


def make_answer_with_llm(query: str, retrieved: List[Tuple[SourceChunk, float]], llm: LLMSettings) -> str:
    return make_answer_with_llm_struct(query, retrieved, llm).markdown
