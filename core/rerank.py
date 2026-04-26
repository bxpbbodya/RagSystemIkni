from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

from .sources import SourceChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


_RERANKER_CACHE: dict[str, Any] = {}


@dataclass
class RerankSettings:
    enabled: bool = False
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    top_n: int = 10
    normalize_scores: bool = True
    alpha: float = 0.9
    batch_size: int = 24
    max_text_chars: int = 1100


def get_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("CrossEncoder not installed. Run: pip install sentence-transformers") from exc

    if model_name not in _RERANKER_CACHE:
        try:
            _RERANKER_CACHE[model_name] = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def _prepare_rerank_text(chunk: SourceChunk, max_chars: int) -> str:
    text = (chunk.text or "").strip()
    title = (chunk.title or "").strip()
    url = (chunk.url or "").strip()
    snippet = text[:max_chars].strip()

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if url:
        parts.append(f"URL: {url}")
    if snippet:
        parts.append(f"TEXT: {snippet}")
    return "\n".join(parts)


def rerank_candidates(
    query: str,
    candidates: List[Tuple[SourceChunk, float]],
    settings: RerankSettings,
    return_scores: bool = True,
) -> List[Tuple[SourceChunk, float]]:
    if not settings.enabled or not candidates:
        return candidates

    reranker = get_reranker(settings.model_name)
    pairs = [
        (query, _prepare_rerank_text(chunk, settings.max_text_chars))
        for chunk, _ in candidates
    ]

    scores = reranker.predict(pairs, batch_size=settings.batch_size)
    scores = np.array(scores, dtype=float)

    if settings.normalize_scores:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    merged: List[Tuple[SourceChunk, float]] = []
    for (chunk, initial_score), rerank_score in zip(candidates, scores):
        combined = settings.alpha * float(rerank_score) + (1.0 - settings.alpha) * float(initial_score)
        merged.append((chunk, combined))

    merged.sort(key=lambda item: item[1], reverse=True)
    if settings.top_n > 0:
        merged = merged[: settings.top_n]

    if not return_scores:
        return [(chunk, 0.0) for chunk, _ in merged]
    return merged


def rerank_results(
    query: str,
    results: List[Tuple[SourceChunk, float]],
    model_name: str,
    top_k: int = 5,
    normalize_scores: bool = True,
    alpha = 0.9,
) -> List[Tuple[SourceChunk, float]]:
    settings = RerankSettings(
        enabled=True,
        model_name=model_name,
        top_n=max(top_k, 1),
        normalize_scores=normalize_scores,
        alpha=alpha,
    )
    ranked = rerank_candidates(query=query, candidates=results, settings=settings)
    return ranked[:top_k]
