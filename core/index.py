# core/index.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .sources import SourceChunk

# -----------------------------
# Model cache (speed-up)
# -----------------------------
_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def get_embed_model(name: str) -> SentenceTransformer:
    """
    Cached SentenceTransformer loader.
    - Uses Streamlit cache if available
    - Falls back to simple global cache otherwise
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _load(n: str) -> SentenceTransformer:
            return SentenceTransformer(n)

        return _load(name)

    except Exception:
        # fallback (non-streamlit environments)
        if name not in _MODEL_CACHE:
            _MODEL_CACHE[name] = SentenceTransformer(name)
        return _MODEL_CACHE[name]



# -----------------------------
# IO
# -----------------------------
def load_chunks_from_jsonl(path: Path) -> List[SourceChunk]:
    chunks: List[SourceChunk] = []
    if not path.exists():
        return chunks

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # in case extra missing
                if obj.get("extra") is None:
                    obj["extra"] = {}
                chunks.append(SourceChunk(**obj))
            except Exception:
                # skip broken line
                continue

    return chunks




def save_chunks_to_jsonl(path: Path, chunks: List[SourceChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")


# -----------------------------
# Index build/load
# -----------------------------
def build_faiss_index(
    chunks: List[SourceChunk],
    embed_model_name: str,
    index_path: Path,
    meta_path: Path,
) -> Tuple[faiss.IndexFlatIP, List[SourceChunk]]:
    """
    Build a cosine-similarity FAISS index using normalized embeddings.
    Uses IndexFlatIP on normalized vectors => cosine similarity.
    """
    if not chunks:
        raise ValueError("No chunks provided to build index.")

    model = get_embed_model(embed_model_name)
    texts = [c.text for c in chunks]

    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

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


# -----------------------------
# Search helpers
# -----------------------------
def _query_keywords(query: str) -> set[str]:
    """
    Extract keywords from query (very simple).
    """
    q = query.lower()
    q = re.sub(r"[^a-zа-яіїєґ0-9\s]", " ", q)
    words = [w.strip() for w in q.split() if len(w.strip()) >= 4]
    return set(words)


def _chunk_has_keywords(chunk_text: str, keywords: set[str], min_hits: int = 1) -> bool:
    if not keywords:
        return True
    text = chunk_text.lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits >= min_hits


def _is_obviously_noise(chunk_text: str) -> bool:
    """
    Filter out obvious garbage:
    - table-like pipes and pluses
    - too many symbols
    """
    t = chunk_text.strip()
    if not t:
        return True
    if len(t) < 60:
        return True
    # lots of pipes/pluses patterns
    if t.count("|") > 25:
        return True
    if t.count("+") > 25:
        return True
    # repetitive " | | + | |"
    if "| |" in t and "+" in t and t.count("| |") > 10:
        return True
    return False

def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[SourceChunk],
    embed_model_name: str,
    top_k: int = 5,
    *,
    min_score: float = 0.35,
    keyword_filter: bool = True,
    internal_k: Optional[int] = None,   # NEW
) -> List[Tuple[SourceChunk, float]]:
    """
    Search FAISS index and apply:
    - min_score threshold
    - optional keyword filter
    - noise filter

    Returns top results (sorted by FAISS score).
    """
    model = get_embed_model(embed_model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    # NEW: allow caller to pass internal_k
    if internal_k is None:
        internal_k = max(top_k * 8, 30)

    scores, ids = index.search(q_emb, internal_k)

    keywords = _query_keywords(query) if keyword_filter else set()

    results: List[Tuple[SourceChunk, float]] = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        score_f = float(score)
        if score_f < min_score:
            continue

        ch = chunks[int(idx)]

        if _is_obviously_noise(ch.text):
            continue

        if keyword_filter and not _chunk_has_keywords(ch.text, keywords, min_hits=1):
            title_url = (ch.title or "") + " " + (ch.url or "")
            if not _chunk_has_keywords(title_url, keywords, min_hits=1):
                continue

        results.append((ch, score_f))

    # we return all candidates (not only top_k) so reranker can use them
    # caller can slice to top_k if reranker disabled
    return results

def filter_results(
    results: List[Tuple[SourceChunk, float]],
    *,
    allowed_types: Optional[set[str]] = None,
    allowed_doc_ids: Optional[set[str]] = None,
) -> List[Tuple[SourceChunk, float]]:
    out = []
    for ch, sc in results:
        stype = (ch.source_type or "").lower()
        if allowed_types and stype not in allowed_types:
            continue

        if allowed_doc_ids:
            extra = ch.extra or {}
            doc_id = extra.get("doc_id")
            if doc_id not in allowed_doc_ids:
                continue

        out.append((ch, sc))
    return out
