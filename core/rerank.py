# core/rerank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .sources import SourceChunk

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


# cache so model loads only once
_RERANKER_CACHE = {}


@dataclass
class RerankSettings:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 10  # how many candidates to keep after reranking


def get_reranker(model_name: str):
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder not installed. Install: pip install sentence-transformers")
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def rerank_candidates(
    query: str,
    candidates: List[Tuple[SourceChunk, float]],
    settings: RerankSettings,
) -> List[Tuple[SourceChunk, float]]:
    """
    candidates: list of (chunk, faiss_score)
    returns: list of (chunk, rerank_score) sorted desc
    """
    if not settings.enabled:
        return candidates

    if not candidates:
        return candidates

    reranker = get_reranker(settings.model_name)

    pairs = [(query, c.text) for c, _ in candidates]
    scores = reranker.predict(pairs)

    # cross-encoder scores are usually not normalized
    scores = np.array(scores, dtype=float)

    ranked = []
    for (chunk, _faiss_score), rr in zip(candidates, scores):
        ranked.append((chunk, float(rr)))

    ranked.sort(key=lambda x: x[1], reverse=True)

    # keep top_n
    if settings.top_n and settings.top_n > 0:
        ranked = ranked[: settings.top_n]

    return ranked

def rerank_results(
    query: str,
    results: List[Tuple[SourceChunk, float]],
    model_name: str,
    top_k: int = 5,
) -> List[Tuple[SourceChunk, float]]:
    """
    Wrapper для сумісності з app.py:
      results = [(chunk, score), ...]  -> повертає топ_k після rerank
    """
    settings = RerankSettings(enabled=True, model_name=model_name, top_n=max(top_k, 1))

    ranked = rerank_candidates(
        query=query,
        candidates=results,
        settings=settings,
    )

    return ranked[:top_k]

