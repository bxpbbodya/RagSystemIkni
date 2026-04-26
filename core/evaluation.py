from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .llm import LLMSettings
from .query_pipeline import answer_query
from .sources import SourceChunk


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    embed_model: str
    top_k: int = 5
    allowed_types: Optional[Tuple[str, ...]] = None
    use_hybrid: bool = True
    use_post_boosts: bool = True
    use_reranker: bool = False
    reranker_model: Optional[str] = None
    reranker_top_n: int = 20
    use_query_expansion: bool = True
    use_adaptive_top_k: bool = True
    use_generation: bool = True
    use_extraction: bool = True
    use_institute_filter: bool = True
    use_rules: bool = True
    keyword_filter: bool = False
    min_score: float = 0.2


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"[^a-zа-яіїєґ0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match_and_f1(prediction: str, example: Dict[str, Any]) -> Tuple[float, float]:
    answer = (example.get("answer") or "").strip()
    if answer:
        pred_norm = normalize_answer(prediction)
        ref_norm = normalize_answer(answer)
        exact = 1.0 if pred_norm == ref_norm and pred_norm else 0.0
        return exact, token_f1(prediction, answer)

    keywords = [normalize_answer(keyword) for keyword in (example.get("answer_keywords") or []) if normalize_answer(keyword)]
    if not keywords:
        return 0.0, 0.0
    pred_norm = normalize_answer(prediction)
    matched = sum(1 for keyword in keywords if keyword in pred_norm)
    exact = 1.0 if matched == len(keywords) else 0.0
    precision = matched / max(len(set(pred_norm.split())), 1)
    recall = matched / len(keywords)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return exact, f1


def is_hit(chunk: SourceChunk, example: Dict[str, Any]) -> bool:
    url = (chunk.url or "").lower()
    title = (chunk.title or "").lower()
    text = (chunk.text or "").lower()
    source_type = (chunk.source_type or "").lower()

    must_url = (example.get("must_contain_url") or "").lower()
    must_type = (example.get("must_contain_type") or "").lower()
    must_text = (example.get("must_contain_text") or "").lower()
    keywords = [str(keyword).lower() for keyword in (example.get("answer_keywords") or [])]

    if must_url and must_url not in url:
        return False
    if must_type and must_type != source_type:
        return False
    if must_text and must_text not in text and must_text not in title:
        return False
    if keywords:
        combined = f"{title} {text}"
        if not any(keyword in combined for keyword in keywords):
            return False
    return True


def run_pipeline_evaluation(
    *,
    eval_set: Sequence[Dict[str, Any]],
    index: Any,
    chunks: List[SourceChunk],
    config: PipelineConfig,
    llm: Optional[LLMSettings] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    hits = 0
    reciprocal_rank_sum = 0.0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    exact_sum = 0.0
    f1_sum = 0.0
    precision_sum = 0.0
    invalid_role_answers = 0
    wrong_institute_answers = 0
    hallucinations = 0

    for example in eval_set:
        query = example.get("query", "")
        answer, bundle = answer_query(
            query=query,
            index=index,
            chunks=chunks,
            embed_model_name=config.embed_model,
            llm=llm if config.use_generation else None,
            min_score=config.min_score,
            keyword_filter=config.keyword_filter,
            allowed_types=set(config.allowed_types) if config.allowed_types else None,
            use_reranker=config.use_reranker,
            reranker_model=config.reranker_model,
            reranker_top_n=config.reranker_top_n,
            use_query_expansion=config.use_query_expansion,
            use_adaptive_top_k=config.use_adaptive_top_k,
            use_hybrid=config.use_hybrid,
            top_k_override=config.top_k,
            use_post_boosts=config.use_post_boosts,
            use_extraction=config.use_extraction,
            use_institute_filter=config.use_institute_filter,
            use_rules=config.use_rules,
        )

        results = bundle.results
        hit_rank = None
        relevant_count = 0
        for rank, (chunk, _) in enumerate(results, start=1):
            if is_hit(chunk, example):
                relevant_count += 1
                if hit_rank is None:
                    hit_rank = rank
        precision_at_k = relevant_count / config.top_k if config.top_k else 0.0
        precision_sum += precision_at_k

        if hit_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / hit_rank
            if hit_rank == 1:
                hits_at_1 += 1
            if hit_rank <= 3:
                hits_at_3 += 1
            if hit_rank <= 5:
                hits_at_5 += 1

        exact_match, f1 = compute_exact_match_and_f1(answer.answer_text, example)
        exact_sum += exact_match
        f1_sum += f1
        pred_norm = normalize_answer(answer.answer_text)
        expected_institute_url = normalize_answer(str(example.get("must_contain_url") or ""))
        if "заступник" in pred_norm or "адміністратор" in pred_norm:
            invalid_role_answers += 1
        if expected_institute_url and expected_institute_url not in normalize_answer(str(results[0][0].url if results else "")):
            wrong_institute_answers += 1
        if answer.answer_text and answer.answer_text != "Не знайдено у базі знань" and hit_rank is None:
            hallucinations += 1

        rows.append(
            {
                "experiment": config.name,
                "query": query,
                "hit": hit_rank is not None,
                "hit_rank": hit_rank,
                "top1_url": results[0][0].url if results else None,
                "top1_score": float(results[0][1]) if results else None,
                "precision_at_k": precision_at_k,
                "answer_text": answer.answer_text,
                "confidence": answer.confidence,
                "exact_match": exact_match,
                "f1": f1,
                "question_type": bundle.analysis.question_type,
                "intent": bundle.analysis.intent,
                "entity_scope": bundle.analysis.entity_scope,
                "expanded_query": bundle.analysis.expanded_query,
                "used_sources": ",".join(str(src) for src in answer.used_sources),
                "entity_candidates": len(bundle.entity_candidates),
            }
        )
        if exact_match < 1.0:
            error_rows.append(
                {
                    "experiment": config.name,
                    "query": query,
                    "expected_answer": example.get("answer", ""),
                    "predicted_answer": answer.answer_text,
                    "intent": bundle.analysis.intent,
                    "entity_scope": bundle.analysis.entity_scope,
                    "hit_at_1": bool(hit_rank == 1),
                    "hit_rank": hit_rank,
                    "top1_url": results[0][0].url if results else None,
                }
            )

    n = len(eval_set)
    details = pd.DataFrame(rows)
    metrics = {
        "experiment": config.name,
        "embed_model": config.embed_model,
        "use_hybrid": config.use_hybrid,
        "use_post_boosts": config.use_post_boosts,
        "allowed_types": ",".join(config.allowed_types) if config.allowed_types else "all",
        "use_reranker": config.use_reranker,
        "use_query_expansion": config.use_query_expansion,
        "use_generation": config.use_generation,
        "use_extraction": config.use_extraction,
        "use_institute_filter": config.use_institute_filter,
        "use_rules": config.use_rules,
        "recall_at_k": hits / n if n else 0.0,
        "mrr_at_k": reciprocal_rank_sum / n if n else 0.0,
        "hit_at_1": hits_at_1 / n if n else 0.0,
        "hit_at_3": hits_at_3 / n if n else 0.0,
        "hit_at_5": hits_at_5 / n if n else 0.0,
        "avg_precision_at_k": precision_sum / n if n else 0.0,
        "exact_match": exact_sum / n if n else 0.0,
        "f1": f1_sum / n if n else 0.0,
        "top1_score_mean": float(details["top1_score"].dropna().mean()) if not details["top1_score"].dropna().empty else None,
        "error_count": len(error_rows),
        "invalid_role_answers": invalid_role_answers,
        "wrong_institute_answers": wrong_institute_answers,
        "hallucinations": hallucinations,
    }
    details.attrs["errors"] = error_rows
    return metrics, details
