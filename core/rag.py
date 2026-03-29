# core/rag.py (optimized for higher metrics)
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .sources import SourceChunk
from .llm import LLMSettings, chat_completion
import heapq

# -----------------------------
# Data models
# -----------------------------
@dataclass
class RAGAnswer:
    markdown: str
    used_sources: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_map: Dict[int, SourceChunk] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

# -----------------------------
# Text utilities
# -----------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)

def _snip(text: str, max_len: int = 320) -> str:
    text = _clean_text(text)
    return text if len(text) <= max_len else text[:max_len].rstrip() + "…"

def _extract_summary(text: str, max_sentences: int = 3, min_length: int = 15) -> str:
    """Краще підсумування: до 3 речень, відкидаємо короткі або повторювані."""
    text = _clean_text(text)
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= min_length]
    return " ".join(sentences[:max_sentences]) if sentences else _snip(text, 250)

def _query_keywords(query: str, min_len: int = 3) -> List[str]:
    words = re.findall(rf"[a-zа-яіїєґ0-9]{{{min_len},}}", query.lower())
    return list(dict.fromkeys(words))[:25]  # більше ключових слів для підсвітки

def highlight_keywords_md(text: str, keywords: List[str]) -> str:
    if not keywords:
        return text
    kws = sorted(set(keywords), key=len, reverse=True)
    pattern = r"(?<!\]\()(" + "|".join(re.escape(k) for k in kws) + r")(?!\))"
    try:
        return re.sub(pattern, lambda m: f"<mark>{m.group(1)}</mark>", text, flags=re.IGNORECASE)
    except Exception:
        return text

# -----------------------------
# Sources formatting
# -----------------------------
def format_sources_md(retrieved: List[Tuple[SourceChunk, float]], top: int = 5) -> str:
    lines = []
    for i, (chunk, score) in enumerate(retrieved[:top], start=1):
        title = _clean_text(chunk.title or "джерело").replace("[", "(").replace("]", ")")
        lines.append(f"[{i}] [{title}]({chunk.url or ''}) — `{chunk.source_type or 'unknown'}` • {chunk.date or ''} • score={score:.3f}")
    return "\n".join(lines)

def build_context(retrieved: List[Tuple[SourceChunk, float]], max_chars: int = 7000, max_chunks: int = 8) -> str:
    """Більший контекст для LLM, включає топ chunks з highest score"""
    ctx_parts, total = [], 0
    top_chunks = heapq.nlargest(max_chunks, retrieved, key=lambda x: x[1])
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        piece = f"[{i}] ({chunk.source_type}) {chunk.title}\nURL: {chunk.url}\nDATE: {chunk.date}\nTEXT: {chunk.text}\n\n"
        if total + len(piece) > max_chars:
            break
        ctx_parts.append(piece)
        total += len(piece)
    return "".join(ctx_parts)

# -----------------------------
# Citation parsing & enforcement
# -----------------------------
def parse_used_sources(text: str, k_max: int) -> List[int]:
    return sorted(set(int(m) for m in re.findall(r"\[(\d{1,2})\]", text) if 1 <= int(m) <= k_max))

def enforce_sources_block(answer: str, used_sources: List[int]) -> str:
    if not used_sources:
        used_str = "немає (контекст недостатній)"
    else:
        used_str = ", ".join(f"[{i}]" for i in used_sources)
    if re.search(r"\bджерела\b\s*:", answer, flags=re.IGNORECASE):
        return answer.strip()
    return f"{answer.strip()}\n\n**Джерела:** {used_str}"

def answer_has_no_data_phrase(text: str) -> bool:
    return any(p in (text or "").lower() for p in ["немає даних у контексті", "недостатньо даних у контексті", "не можу знайти у контексті"])

# -----------------------------
# Offline answer (optimized)
# -----------------------------
def make_answer_no_llm_struct(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> RAGAnswer:
    if not retrieved:
        return RAGAnswer(markdown=f"**Запит:** {query}\n\n❌ **Нічого не знайдено.**", warnings=["no_results"])

    bullets, seen_norm, used_sources = [], set(), set()
    for idx, (chunk, _) in enumerate(retrieved[:8], start=1):
        txt = _extract_summary(chunk.text)
        norm = txt.lower() if txt else ""
        if txt and norm not in seen_norm:
            bullets.append(f"- {txt} **[{idx}]**")
            seen_norm.add(norm)
            used_sources.add(idx)
        if len(bullets) >= 5:
            break

    sources_md = format_sources_md(retrieved, top=10)
    md = f"**Запит:** {query}\n\n### ✅ Відповідь (offline)\n" \
         f"{chr(10).join(bullets) if bullets else '_Немає тексту для підсумку._'}\n\n---\n" \
         f"### 📌 Джерела\n{sources_md}"
    md = highlight_keywords_md(md, _query_keywords(query))
    source_map = {i: retrieved[i - 1][0] for i in used_sources if 1 <= i <= len(retrieved)}
    metrics = {"num_bullets": len(bullets), "num_used_sources": len(used_sources), "retrieved_count": len(retrieved)}

    return RAGAnswer(markdown=md, used_sources=sorted(used_sources), source_map=source_map, metrics=metrics)

# -----------------------------
# LLM answer (optimized)
# -----------------------------
def make_answer_with_llm_struct(query: str, retrieved: List[Tuple[SourceChunk, float]], llm: LLMSettings) -> RAGAnswer:
    if not retrieved:
        return make_answer_no_llm_struct(query, retrieved)

    k_max = min(len(retrieved), 8)
    context = build_context(retrieved, max_chars=9000, max_chunks=k_max)

    system_prompt = (
        "Ти — помічник для студентів ЛПНУ.\n"
        "Відповідай українською.\n"
        "Використовуй лише надану інформацію.\n"
        "Не вигадуй даних.\n"
        "Після кожного факту став посилання на джерело [1], [2], ...\n"
    )
    user_prompt = f"Питання: {query}\n\nКонтекст:\n{context}\n\n" \
                  "Відповідь у 5–10 коротких пунктів з посиланнями [n].\n" \
                  "Наприкінці напиши 'Джерела: [..]' тільки використані."

    warnings: List[str] = []
    try:
        content = chat_completion(llm, messages=[{"role": "system", "content": system_prompt},
                                                 {"role": "user", "content": user_prompt}])
    except Exception as e:
        warnings.append(f"llm_error: {e}")
        fallback = make_answer_no_llm_struct(query, retrieved)
        fallback.warnings += warnings
        return fallback

    used = parse_used_sources(content, k_max=k_max)
    if not used:
        warnings.append("no_citations_found")
    content = enforce_sources_block(content, used)
    if answer_has_no_data_phrase(content):
        warnings.append("llm_says_no_data")

    sources_md = format_sources_md(retrieved, top=8)
    md = f"**Запит:** {query}\n\n### ✅ Відповідь (LLM)\n{content.strip()}\n\n---\n" \
         f"### 📌 Джерела (retrieval топ-8)\n{sources_md}"

    md = highlight_keywords_md(md, _query_keywords(query))
    source_map = {i: retrieved[i - 1][0] for i in used if 1 <= i <= len(retrieved)}
    metrics = {"num_used_sources": len(used), "num_retrieved": len(retrieved), "llm_warning_count": len(warnings)}

    return RAGAnswer(markdown=md, used_sources=used, warnings=warnings, source_map=source_map, metrics=metrics)

# -----------------------------
# Simple wrappers
# -----------------------------
def make_answer_no_llm(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> str:
    return make_answer_no_llm_struct(query, retrieved).markdown

def make_answer_with_llm(query: str, retrieved: List[Tuple[SourceChunk, float]], llm: LLMSettings) -> str:
    return make_answer_with_llm_struct(query, retrieved, llm).markdown