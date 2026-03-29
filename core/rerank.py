# core/rerank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .sources import SourceChunk

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# -----------------------------
# Cache for loaded models
# -----------------------------
_RERANKER_CACHE: dict[str, CrossEncoder] = {}

# -----------------------------
# Settings dataclass
# -----------------------------
@dataclass
class RerankSettings:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 10
    normalize_scores: bool = True  # нормалізувати reranker score
    alpha: float = 0.7             # вага reranker score
    batch_size: int = 32            # batch size для predict


# -----------------------------
# Model loader
# -----------------------------
def get_reranker(model_name: str) -> CrossEncoder:
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder not installed. Run: pip install sentence-transformers")
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


# -----------------------------
# Reranking logic
# -----------------------------
def rerank_candidates(
        query: str,
        candidates: List[Tuple[SourceChunk, float]],
        settings: RerankSettings,
        return_scores: bool = True,
) -> List[Tuple[SourceChunk, float]]:
    """
    Rerank candidates using CrossEncoder + initial score.
    """
    if not settings.enabled or not candidates:
        return candidates

    reranker = get_reranker(settings.model_name)
    pairs = [(query, c.text) for c, _ in candidates]

    # передбачення батчами
    scores = reranker.predict(pairs, batch_size=settings.batch_size)
    scores = np.array(scores, dtype=float)

    # normalize reranker score
    if settings.normalize_scores:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # змішування reranker score та initial score
    final_scores = []
    for (chunk, init_score), rerank_score in zip(candidates, scores):
        combined = settings.alpha * rerank_score + (1 - settings.alpha) * init_score
        final_scores.append((chunk, float(combined)))

    # сортування і top_n
    final_scores.sort(key=lambda x: x[1], reverse=True)
    if settings.top_n > 0:
        final_scores = final_scores[:settings.top_n]

    if not return_scores:
        final_scores = [(chunk, 0.0) for chunk, _ in final_scores]

    return final_scores


# -----------------------------
# Wrapper для app.py
# -----------------------------
def rerank_results(
        query: str,
        results: List[Tuple[SourceChunk, float]],
        model_name: str,
        top_k: int = 5,
        normalize_scores: bool = True,
        alpha: float = 0.7,
) -> List[Tuple[SourceChunk, float]]:
    """
    Simple wrapper: rerank results and return top_k
    """
    settings = RerankSettings(
        enabled=True,
        model_name=model_name,
        top_n=max(top_k, 1),
        normalize_scores=normalize_scores,
        alpha=alpha,
    )
    ranked = rerank_candidates(query=query, candidates=results, settings=settings)
    return ranked[:top_k]