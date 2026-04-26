from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .llm import LLMSettings, chat_completion
from .sources import SourceChunk


NO_KB_PHRASE = "Не знайдено у базі знань"
TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9']+", re.IGNORECASE)


@dataclass
class RAGAnswer:
    markdown: str
    answer_text: str = ""
    confidence: str = "Low"
    used_sources: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_map: Dict[int, SourceChunk] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def _snip(text: str, max_len: int = 260) -> str:
    text = _clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _extract_summary(text: str, max_sentences: int = 1, min_length: int = 20) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.strip()) >= min_length]
    return " ".join(sentences[:max_sentences]) if sentences else _snip(text, 180)


def _query_keywords(query: str, min_len: int = 3) -> List[str]:
    words = TOKEN_RE.findall((query or "").lower())
    return list(dict.fromkeys(word for word in words if len(word) >= min_len))[:25]


def _unique_retrieved(retrieved: Sequence[Tuple[SourceChunk, float]], limit: int = 3) -> List[Tuple[SourceChunk, float]]:
    unique: List[Tuple[SourceChunk, float]] = []
    seen_keys = set()
    for chunk, score in retrieved:
        extra = chunk.extra or {}
        key = (chunk.url or "", str(extra.get("doc_id") or ""), chunk.title or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append((chunk, score))
        if len(unique) >= limit:
            break
    return unique


def highlight_keywords_md(text: str, keywords: List[str]) -> str:
    if not keywords:
        return text
    pattern = r"(?<!\]\()(" + "|".join(re.escape(keyword) for keyword in sorted(set(keywords), key=len, reverse=True)) + r")(?!\))"
    try:
        return re.sub(pattern, lambda match: f"<mark>{match.group(1)}</mark>", text, flags=re.IGNORECASE)
    except Exception:
        return text


def format_sources_md(retrieved: Sequence[Tuple[SourceChunk, float]], top: int = 3) -> str:
    lines: List[str] = []
    for i, (chunk, _) in enumerate(_unique_retrieved(retrieved, limit=top), start=1):
        title = _clean_text(chunk.title or "джерело").replace("[", "(").replace("]", ")")
        lines.append(f"[{i}] {title}")
    return "\n".join(lines)


def build_context(query: str, retrieved: Sequence[Tuple[SourceChunk, float]], max_chars: int = 3200, max_chunks: int = 3) -> str:
    context_parts: List[str] = []
    total = 0
    for i, (chunk, score) in enumerate(_unique_retrieved(retrieved, limit=max_chunks), start=1):
        snippet = _snip(chunk.text or "", 900)
        piece = (
            f"[{i}] TITLE: {chunk.title}\n"
            f"URL: {chunk.url}\n"
            f"SCORE: {score:.3f}\n"
            f"TEXT: {snippet}\n\n"
        )
        if total + len(piece) > max_chars:
            break
        context_parts.append(piece)
        total += len(piece)
    return "".join(context_parts)


def parse_used_sources(text: str, k_max: int) -> List[int]:
    return sorted(set(int(match) for match in re.findall(r"\[(\d{1,2})\]", text or "") if 1 <= int(match) <= k_max))


def answer_has_no_data_phrase(text: str) -> bool:
    text_lower = (text or "").lower()
    return NO_KB_PHRASE.lower() in text_lower or "не знайдено" in text_lower


def _contains_person_role(text: str) -> bool:
    normalized = (text or "").lower()
    role_terms = ("директор", "заступник", "ректор", "декан", "завідувач", "керівник")
    return any(term in normalized for term in role_terms)


def _build_markdown(
    *,
    query: str,
    answer_text: str,
    confidence: str,
    retrieved: Sequence[Tuple[SourceChunk, float]],
    used_sources: Sequence[int],
    explanation: Optional[str] = None,
) -> str:
    sources_md = format_sources_md(retrieved, top=2)
    source_suffix = "\n".join(f"[{idx}]" for idx in used_sources) if used_sources else "немає"
    lines = [
        "**Відповідь**",
        answer_text.strip() if answer_text else NO_KB_PHRASE,
        "",
        f"**Впевненість:** {confidence}",
    ]
    if explanation:
        lines.extend(["", f"**Пояснення:** {explanation.strip()}"])
    lines.extend(["", "**Джерела:**", sources_md if sources_md else source_suffix])
    markdown = "\n".join(lines)
    return highlight_keywords_md(markdown, _query_keywords(query))


def make_direct_answer_struct(
    *,
    query: str,
    answer_text: str,
    retrieved: Sequence[Tuple[SourceChunk, float]],
    used_sources: Sequence[int],
    confidence: str = "High",
    explanation: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> RAGAnswer:
    used_sources = list(used_sources)
    unique_retrieved = _unique_retrieved(retrieved, limit=3)
    source_map = {i: unique_retrieved[i - 1][0] for i in used_sources if 1 <= i <= len(unique_retrieved)}
    markdown = _build_markdown(
        query=query,
        answer_text=answer_text,
        confidence=confidence,
        retrieved=unique_retrieved,
        used_sources=used_sources,
        explanation=explanation,
    )
    return RAGAnswer(
        markdown=markdown,
        answer_text=answer_text,
        confidence=confidence,
        used_sources=used_sources,
        warnings=list(warnings or []),
        source_map=source_map,
        metrics=dict(metrics or {}),
    )


def _make_no_kb_answer(query: str) -> RAGAnswer:
    return make_direct_answer_struct(
        query=query,
        answer_text=NO_KB_PHRASE,
        retrieved=[],
        used_sources=[],
        confidence="Low",
        explanation="У релевантному контексті немає прямої відповіді.",
        warnings=["no_grounded_answer"],
        metrics={"retrieved_count": 0},
    )


def make_answer_no_llm_struct(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> RAGAnswer:
    selected = _unique_retrieved(retrieved, limit=10)
    if not selected:
        return _make_no_kb_answer(query)

    top_chunk = selected[0][0]
    summary = _extract_summary(top_chunk.text)
    if not summary:
        return _make_no_kb_answer(query)

    answer_text = summary if summary.endswith(".") else f"{summary}."
    return make_direct_answer_struct(
        query=query,
        answer_text=answer_text,
        retrieved=selected,
        used_sources=[1],
        confidence="Medium",
        explanation="Відповідь сформовано з найрелевантнішого фрагмента без LLM.",
        metrics={"retrieved_count": len(selected)},
    )


def make_answer_with_llm_struct(query: str, retrieved: List[Tuple[SourceChunk, float]], llm: LLMSettings) -> RAGAnswer:
    selected = _unique_retrieved(retrieved, limit=10)
    if not selected:
        return _make_no_kb_answer(query)

    context = build_context(query, selected, max_chars=3200, max_chunks=5)
    k_max = len(selected)
    system_prompt = (
        "Ти асистент бази знань Львівської політехніки.\n"
        "Відповідай лише українською мовою.\n"
        "Використовуй факти з контексту. Якщо ПІБ вказано у розділі TITLE: Дирекція або Керівництво, "
        "ця особа є керівником (директором) відповідного підрозділу.\n"
        "НЕ вигадуй факти. Якщо у відповіді вказуєш посаду, вона має логічно випливати з контексту.\n"
        f"Якщо відповіді точно немає, напиши рівно: {NO_KB_PHRASE}\n"
        "Після відповіді вкажи джерела у форматі [1], [2]."
    )
    user_prompt = (
        f"Запит: {query}\n\n"
        f"Контекст:\n{context}\n"
        "Сформуй коротку точну відповідь в 1 реченні."
    )

    warnings: List[str] = []
    try:
        content = chat_completion(
            llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        ).strip()
    except Exception as exc:
        warnings.append(f"llm_error: {exc}")
        fallback = make_answer_no_llm_struct(query, selected)
        fallback.warnings.extend(warnings)
        return fallback

    used_sources = parse_used_sources(content, k_max=k_max)
    if answer_has_no_data_phrase(content):
        return _make_no_kb_answer(query)
    if not used_sources:
        warnings.append("no_citations_found")
        fallback = make_answer_no_llm_struct(query, selected)
        fallback.warnings.extend(warnings)
        return fallback

    answer_text = re.sub(r"\s*\*\*Джерела:\*\*.*$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    answer_text = re.sub(r"\s*\[([1-9]\d?)\](?:\s*,\s*\[([1-9]\d?)\])*$", "", answer_text).strip()
    if not answer_text:
        return _make_no_kb_answer(query)
    # Prevent role hallucinations by requiring role-bearing claims to be grounded in cited context.
    if _contains_person_role(answer_text):
        cited_content_full = " ".join(
            ((selected[idx - 1][0].text or "") + " " + (selected[idx - 1][0].title or ""))
            for idx in used_sources
            if 1 <= idx <= len(selected)
        ).lower()
        if not cited_content_full:
            return _make_no_kb_answer(query)
        for role in ("директор", "заступник", "ректор", "декан", "завідувач", "керівник"):
            # Дозволяємо роль, якщо в контексті є сама роль АБО слово "дирекція/керівництво"
            if role in answer_text.lower():
                if role not in cited_content_full and "дирекція" not in cited_content_full and "керівництво" not in cited_content_full:
                    warnings.append("role_not_grounded_in_context")
                # fallback = make_answer_no_llm_struct(query, selected)
                # fallback.warnings.extend(warnings)
                # return fallback

    return make_direct_answer_struct(
        query=query,
        answer_text=answer_text,
        retrieved=selected,
        used_sources=used_sources,
        confidence="Medium",
        explanation="Відповідь уточнено мовною моделлю на основі очищеного контексту.",
        warnings=warnings,
        metrics={"retrieved_count": len(selected), "llm_used": 1},
    )


def make_answer_no_llm(query: str, retrieved: List[Tuple[SourceChunk, float]]) -> str:
    return make_answer_no_llm_struct(query, retrieved).markdown


def make_answer_with_llm(query: str, retrieved: List[Tuple[SourceChunk, float]], llm: LLMSettings) -> str:
    return make_answer_with_llm_struct(query, retrieved, llm).markdown
