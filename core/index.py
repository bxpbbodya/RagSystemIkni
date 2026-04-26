from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np

from .sources import SourceChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9']+", re.IGNORECASE)


INSTITUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "iard": ("іард", "iard", "архітектури", "дизайну"),
    "ikni": ("ікні", "ikni", "комп'ютерних наук", "компютерних наук", "інформаційних технологій"),
    "igdg": ("ігдг", "ігдз", "ihdh", "igdg", "геодез", "землеустр"),
    "ihsn": ("ігсн", "ihsn", "гуманітарн", "соціальн"),
    "iadu": ("іаду", "iadu", "адміністрування", "державного управління"),
    "inem": ("інем", "inem", "економіки", "менеджменту"),
    "ibib": ("ібіб", "ibib", "будівництва", "інфраструктури", "безпеки життєдіяльності"),
}

TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "kerivnytstvo": ("керівництво", "директор", "завідувач", "очолює", "керує"),
    "spetsialnosti": ("спеціальності", "освітні програми", "освітня програма", "напрями підготовки", "програми"),
    "vstupnyku": ("вступ", "вступнику", "подача документів", "документи", "вступні випробування"),
    "kontakty": ("контакти", "зв'язатися", "телефон", "email", "e-mail"),
    "studentske-mistechko": ("гуртожит", "поселення", "проживання"),
    "studentska-biblioteka": ("бібліотек", "книг", "журнал", "електронні ресурси"),
    "news": ("новини", "події", "останні"),
    "partnery": ("партнер", "співпраця", "угоди"),
}


@dataclass
class _BM25Corpus:
    postings: dict[str, list[tuple[int, int]]]
    doc_lengths: list[int]
    avgdl: float
    doc_count: int


_MODEL_CACHE: dict[str, Any] = {}
_BM25_CACHE: dict[tuple[Any, ...], _BM25Corpus] = {}


def _import_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def get_embed_model(name: str):
    SentenceTransformer = _import_sentence_transformer()

    def _construct(model_name: str):
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(model_name)

    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _load(n: str):
            return _construct(n)

        return _load(name)
    except Exception:
        if name not in _MODEL_CACHE:
            _MODEL_CACHE[name] = _construct(name)
        return _MODEL_CACHE[name]


def load_chunks_from_jsonl(path: Path) -> List[SourceChunk]:
    chunks: List[SourceChunk] = []
    if not path.exists():
        return chunks

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("extra") is None:
                    obj["extra"] = {}
                chunks.append(SourceChunk(**obj))
            except Exception:
                continue
    return chunks


def save_chunks_to_jsonl(path: Path, chunks: List[SourceChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def build_faiss_index(
    chunks: List[SourceChunk],
    embed_model_name: str,
    index_path: Path,
    meta_path: Path,
    normalize: bool = True,
) -> Tuple[faiss.IndexFlatIP, List[SourceChunk]]:
    if not chunks:
        raise ValueError("No chunks provided to build index.")

    model = get_embed_model(embed_model_name)
    texts = [chunk.text for chunk in chunks]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    save_chunks_to_jsonl(meta_path, chunks)
    return index, chunks


def load_faiss_index(index_path: Path, meta_path: Path) -> Tuple[faiss.IndexFlatIP, List[SourceChunk]]:
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("FAISS index or metadata not found. Build index first.")
    index = faiss.read_index(str(index_path))
    chunks = load_chunks_from_jsonl(meta_path)
    return index, chunks


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-zа-яіїєґ0-9/\-_' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").split("#")[0].split("?")[0]


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize_text(text))


def _query_keywords(query: str) -> Set[str]:
    return {token for token in _tokenize(query) if len(token) >= 3}


def _detect_query_institute(query: str) -> Optional[str]:
    q = _normalize_text(query)
    for code, aliases in INSTITUTE_ALIASES.items():
        if any(alias in q for alias in aliases):
            return code
    return None


def _detect_query_topics(query: str) -> Set[str]:
    q = _normalize_text(query)
    matched: Set[str] = set()
    for topic, hints in TOPIC_HINTS.items():
        if any(hint in q for hint in hints):
            matched.add(topic)
    return matched


def _chunk_search_text(chunk: SourceChunk) -> str:
    extra = chunk.extra or {}
    parts = [
        chunk.title or "",
        chunk.url or "",
        chunk.source_type or "",
        extra.get("doc_id") or "",
        (chunk.text or "")[:1500],
    ]
    return _normalize_text(" \n ".join(part for part in parts if part))


def _chunk_matches_institute(query: str, chunk: SourceChunk) -> bool:
    code = _detect_query_institute(query)
    if not code:
        return True
    haystack = _chunk_search_text(chunk)
    return any(alias in haystack for alias in INSTITUTE_ALIASES.get(code, ()))


def _chunk_matches_topics(query: str, chunk: SourceChunk) -> bool:
    topics = _detect_query_topics(query)
    if not topics:
        return True
    haystack = _normalize_text(f"{chunk.url or ''} {chunk.title or ''}")
    return any(topic in haystack for topic in topics)


def _chunk_has_keywords(chunk: SourceChunk, keywords: Set[str], min_hits: int = 1) -> bool:
    if not keywords:
        return True
    haystack = _chunk_search_text(chunk)
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return hits >= min_hits


def _is_obviously_noise(chunk_text: str) -> bool:
    text = (chunk_text or "").strip()
    if not text or len(text) < 60:
        return True
    if text.count("|") > 25 or text.count("+") > 25:
        return True
    if "| |" in text and "+" in text and text.count("| |") > 10:
        return True
    if re.search(r"([a-z0-9]{3,})\1{5,}", text.lower()):
        return True
    return False


def _topic_path_bonus(query: str, url: str) -> float:
    q = _normalize_text(query)
    u = _normalize_text(url)
    bonus = 0.0
    for topic, hints in TOPIC_HINTS.items():
        if topic in u and any(hint in q for hint in hints):
            bonus += 0.18
    return bonus


def _institute_bonus(query: str, chunk: SourceChunk) -> float:
    code = _detect_query_institute(query)
    if not code:
        return 0.0
    haystack = _chunk_search_text(chunk)
    aliases = INSTITUTE_ALIASES.get(code, ())
    return 0.35 if any(alias in haystack for alias in aliases) else -0.10


def _lexical_bonus(query: str, chunk: SourceChunk) -> float:
    keywords = _query_keywords(query)
    if not keywords:
        return 0.0

    title = _normalize_text(chunk.title or "")
    url = _normalize_text(chunk.url or "")
    text_head = _normalize_text((chunk.text or "")[:400])

    title_hits = sum(1 for keyword in keywords if keyword in title)
    url_hits = sum(1 for keyword in keywords if keyword in url)
    text_hits = sum(1 for keyword in keywords if keyword in text_head)

    bonus = min(title_hits * 0.08, 0.24)
    bonus += min(url_hits * 0.10, 0.30)
    bonus += min(text_hits * 0.025, 0.10)
    bonus += _topic_path_bonus(query, chunk.url or "")
    return bonus


def _chunks_fingerprint(chunks: Iterable[SourceChunk]) -> tuple[Any, ...]:
    chunk_list = list(chunks)
    if not chunk_list:
        return (0,)
    sample = tuple(chunk.chunk_id for chunk in chunk_list[:8])
    return (len(chunk_list), chunk_list[0].chunk_id, chunk_list[-1].chunk_id, sample)


def _get_bm25_corpus(chunks: List[SourceChunk]) -> _BM25Corpus:
    key = _chunks_fingerprint(chunks)
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        return cached

    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: list[int] = []

    for idx, chunk in enumerate(chunks):
        tokens = _tokenize(_chunk_search_text(chunk))
        token_counts = Counter(tokens)
        doc_lengths.append(sum(token_counts.values()))
        for token, term_freq in token_counts.items():
            postings[token].append((idx, term_freq))

    avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    corpus = _BM25Corpus(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        avgdl=avgdl or 1.0,
        doc_count=len(chunks),
    )
    _BM25_CACHE[key] = corpus
    return corpus


def _search_bm25(
    query: str,
    chunks: List[SourceChunk],
    top_n: int,
    *,
    k1: float = 1.6,
    b: float = 0.75,
) -> Dict[int, float]:
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return {}

    corpus = _get_bm25_corpus(chunks)
    token_counts = Counter(query_tokens)
    scores: dict[int, float] = defaultdict(float)

    for token, query_tf in token_counts.items():
        postings = corpus.postings.get(token)
        if not postings:
            continue
        doc_freq = len(postings)
        idf = math.log(1.0 + (corpus.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
        for doc_id, term_freq in postings:
            doc_len = corpus.doc_lengths[doc_id] or 1
            denom = term_freq + k1 * (1.0 - b + b * doc_len / corpus.avgdl)
            bm25 = idf * ((term_freq * (k1 + 1.0)) / max(denom, 1e-8))
            if query_tf > 1:
                bm25 *= 1.0 + 0.08 * (query_tf - 1)
            scores[doc_id] += bm25

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return dict(ranked[:top_n])


def _normalize_score_dict(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    numbers = list(values.values())
    min_value = min(numbers)
    max_value = max(numbers)
    if math.isclose(max_value, min_value):
        if math.isclose(max_value, 0.0):
            return {key: 0.0 for key in values}
        return {key: 1.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def _semantic_component(raw_score: float, normalized_score: float) -> float:
    clipped = max(float(raw_score), 0.0)
    return 0.55 * clipped + 0.45 * normalized_score


def _prefer_matches(
    results: List[Tuple[SourceChunk, float]],
    predicate,
    *,
    min_keep: int,
) -> List[Tuple[SourceChunk, float]]:
    matches = [item for item in results if predicate(item[0])]
    if len(matches) >= min_keep:
        return matches
    if not matches:
        return results
    non_matches = [item for item in results if not predicate(item[0])]
    return matches + non_matches


def _dedupe_results(results: List[Tuple[SourceChunk, float]]) -> List[Tuple[SourceChunk, float]]:
    unique: List[Tuple[SourceChunk, float]] = []
    seen_keys: Set[str] = set()

    for chunk, score in results:
        extra = chunk.extra or {}
        key = _normalize_url(chunk.url or "") or str(extra.get("doc_id") or "") or chunk.chunk_id
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append((chunk, score))
    return unique


def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[SourceChunk],
    embed_model_name: str,
    top_k: int = 5,
    *,
    min_score: float = 0.35,
    keyword_filter: bool = True,
    internal_k: Optional[int] = None,
    use_hybrid: bool = True,
    semantic_weight: float = 0.68,
    bm25_weight: float = 0.32,
    use_query_boosts: bool = True,
) -> List[Tuple[SourceChunk, float]]:
    if internal_k is None:
        internal_k = max(top_k * 20, 120)

    model = get_embed_model(embed_model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    raw_scores, raw_ids = index.search(query_embedding, internal_k)
    semantic_raw: Dict[int, float] = {}
    for idx, score in zip(raw_ids[0], raw_scores[0]):
        if idx == -1:
            continue
        semantic_raw[int(idx)] = float(score)

    bm25_raw = _search_bm25(query=query, chunks=chunks, top_n=internal_k) if use_hybrid else {}
    semantic_norm = _normalize_score_dict(semantic_raw)
    bm25_norm = _normalize_score_dict(bm25_raw)
    keywords = _query_keywords(query) if keyword_filter else set()

    candidate_ids = set()
    for idx, score in semantic_raw.items():
        if score >= min_score or idx in bm25_raw:
            candidate_ids.add(idx)
    candidate_ids.update(bm25_raw.keys())

    results: List[Tuple[SourceChunk, float]] = []
    for idx in candidate_ids:
        chunk = chunks[idx]
        if _is_obviously_noise(chunk.text):
            continue

        if keyword_filter and not _chunk_has_keywords(chunk, keywords, min_hits=1):
            continue

        semantic_score = _semantic_component(semantic_raw.get(idx, 0.0), semantic_norm.get(idx, 0.0))
        bm25_score = bm25_norm.get(idx, 0.0)
        combined = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
        if use_query_boosts:
            combined += _institute_bonus(query, chunk)
            combined += _lexical_bonus(query, chunk)
        results.append((chunk, float(combined)))

    results.sort(key=lambda item: item[1], reverse=True)
    if use_query_boosts:
        results = _prefer_matches(results, lambda chunk: _chunk_matches_institute(query, chunk), min_keep=top_k)
        results = _prefer_matches(results, lambda chunk: _chunk_matches_topics(query, chunk), min_keep=top_k)
    results = _dedupe_results(results)
    return results[: max(internal_k, top_k)]


def filter_results(
    results: List[Tuple[SourceChunk, float]],
    *,
    allowed_types: Optional[Set[str]] = None,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> List[Tuple[SourceChunk, float]]:
    filtered: List[Tuple[SourceChunk, float]] = []
    for chunk, score in results:
        source_type = (chunk.source_type or "").lower()
        if allowed_types and source_type not in allowed_types:
            continue
        if allowed_doc_ids:
            doc_id = (chunk.extra or {}).get("doc_id")
            if doc_id not in allowed_doc_ids:
                continue
        filtered.append((chunk, score))
    return filtered
