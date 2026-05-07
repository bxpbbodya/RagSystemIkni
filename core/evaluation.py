from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .llm import LLMSettings
from .query_pipeline import (
    AnswerQueryConfig,
    analyze_query_with_options,
    answer_query_with_config,
    extract_answer_before_llm,
    source_trust_score,
)
from .rag import NO_KB_PHRASE, make_answer_no_llm_struct
from .sources import SourceChunk

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None


URL_ALIAS_REWRITES: Tuple[Tuple[str, str], ...] = (
    ("lpnu.ua/ihdh/", "lpnu.ua/igdg/"),
    ("lpnu.ua/igdg/", "lpnu.ua/ihdh/"),
)


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
    use_generation: bool = False
    use_extraction: bool = True
    use_institute_filter: bool = True
    use_rules: bool = True
    keyword_filter: bool = True
    min_score: float = 0.15

    def to_answer_query_config(self, llm: Optional[LLMSettings] = None) -> AnswerQueryConfig:
        return AnswerQueryConfig(
            embed_model_name=self.embed_model,
            min_score=self.min_score,
            keyword_filter=self.keyword_filter,
            allowed_types=self.allowed_types,
            use_reranker=self.use_reranker,
            reranker_model=self.reranker_model,
            reranker_top_n=self.reranker_top_n,
            use_query_expansion=self.use_query_expansion,
            use_adaptive_top_k=self.use_adaptive_top_k,
            use_hybrid=self.use_hybrid,
            top_k_override=self.top_k,
            use_post_boosts=self.use_post_boosts,
            use_extraction=self.use_extraction,
            use_institute_filter=self.use_institute_filter,
            use_rules=self.use_rules,
            llm=llm if (llm and self.use_generation) else None,
        )


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


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.lower().strip().rstrip("/")
    url = re.sub(r"https?://(www\.)?", "", url)
    url = url.split("#")[0].split("?")[0]
    return url


def url_variants(url: str) -> List[str]:
    normalized = normalize_url(url)
    if not normalized:
        return []
    variants = {normalized}
    for src, dst in URL_ALIAS_REWRITES:
        if src in normalized:
            variants.add(normalized.replace(src, dst))
    return sorted(variants)


def urls_match(left: str, right: str) -> bool:
    left_variants = url_variants(left)
    right_variants = url_variants(right)
    if not left_variants or not right_variants:
        return False
    return any(
        candidate == expected or candidate.endswith(expected) or expected.endswith(candidate)
        for candidate in left_variants
        for expected in right_variants
    )


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"[^a-zа-щьюяіїєґ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = 0
    ref_counts: Dict[str, int] = {}
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


def compute_exact_match_and_f1(prediction: str, reference: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    expected = (reference or "").strip()
    if not expected:
        return None, None

    p_norm = normalize_answer(prediction)
    e_norm = normalize_answer(expected)
    exact = 1.0 if p_norm == e_norm and p_norm else 0.0
    f1 = token_f1(prediction, expected)
    return exact, f1


def rouge_l_score(prediction: str, reference: Optional[str]) -> Optional[float]:
    if not prediction or not reference or rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def _load_semantic_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return None


_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}


def semantic_similarity(prediction: str, reference: Optional[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> Optional[float]:
    if not prediction or not reference:
        return None
    model = _SEMANTIC_MODEL_CACHE.get(model_name)
    if model is None:
        model = _load_semantic_model(model_name)
        _SEMANTIC_MODEL_CACHE[model_name] = model
    if model is None:
        return None
    try:
        embeddings = model.encode([prediction, reference], normalize_embeddings=True)
        left, right = embeddings
        return float(sum(float(a) * float(b) for a, b in zip(left, right)))
    except Exception:
        return None


def bertscore_f1(prediction: str, reference: Optional[str]) -> Optional[float]:
    if not prediction or not reference or bert_score is None:
        return None
    try:
        _, _, f1 = bert_score([prediction], [reference], lang="uk", verbose=False)
        return float(f1[0].item())
    except Exception:
        return None


def _dcg(relevances: List[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        score += rel / math.log2(idx + 1)
    return score


def ndcg_at_k(results: Sequence[Tuple[SourceChunk, float]], example: Dict[str, Any], *, top_k: int, gold_total: int) -> float:
    if top_k <= 0 or gold_total <= 0:
        return 0.0
    relevances = [1 if is_hit(chunk, example) else 0 for chunk, _ in list(results)[:top_k]]
    dcg = _dcg(relevances)
    ideal_relevances = [1] * min(gold_total, top_k)
    idcg = _dcg(ideal_relevances)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(results: Sequence[Tuple[SourceChunk, float]], example: Dict[str, Any], *, top_k: int, gold_total: int) -> float:
    if top_k <= 0 or gold_total <= 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, (chunk, _) in enumerate(list(results)[:top_k], start=1):
        if not is_hit(chunk, example):
            continue
        hit_count += 1
        precision_sum += hit_count / rank
    if hit_count == 0:
        return 0.0
    return precision_sum / min(gold_total, top_k)


def groundedness_score(answer_text: str, supporting_chunks: Sequence[SourceChunk]) -> Optional[float]:
    answer_tokens = [token for token in normalize_answer(answer_text).split() if len(token) >= 3]
    if not answer_tokens:
        return None
    context = normalize_answer(" ".join(f"{chunk.title} {chunk.text}" for chunk in supporting_chunks))
    if not context:
        return None
    grounded = sum(1 for token in answer_tokens if token in context)
    return grounded / len(answer_tokens)


def trusted_source_ratio(chunks: Sequence[SourceChunk], threshold: float = 0.85) -> Optional[float]:
    if not chunks:
        return None
    trusted = sum(1 for chunk in chunks if source_trust_score(chunk) >= threshold)
    return trusted / len(chunks)


def mean_source_confidence(chunks: Sequence[SourceChunk]) -> Optional[float]:
    if not chunks:
        return None
    return sum(source_trust_score(chunk) for chunk in chunks) / len(chunks)


def citation_correctness(answer, supporting_chunks: Sequence[SourceChunk]) -> Optional[float]:
    if not supporting_chunks:
        return None
    if not answer.used_sources:
        return 0.0
    valid = sum(1 for src in answer.used_sources if 1 <= int(src) <= len(supporting_chunks))
    return valid / len(answer.used_sources)


def estimate_token_usage(query: str, answer_text: str, supporting_chunks: Sequence[SourceChunk]) -> Tuple[int, int]:
    prompt_text = " ".join([query] + [chunk.text for chunk in supporting_chunks])
    prompt_tokens = max(1, int(len(prompt_text.split()) * 1.25))
    completion_tokens = max(1, int(len((answer_text or "").split()) * 1.25)) if answer_text else 0
    return prompt_tokens, completion_tokens


def estimate_cost_usd(llm: Optional[LLMSettings], prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    if not llm or not llm.enabled:
        return None
    price_table = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }
    input_rate, output_rate = price_table.get(llm.model, (0.0, 0.0))
    if input_rate == 0.0 and output_rate == 0.0:
        return None
    return ((prompt_tokens / 1_000_000) * input_rate) + ((completion_tokens / 1_000_000) * output_rate)


def is_hit(chunk: SourceChunk, example: Dict[str, Any]) -> bool:
    must_url = example.get("must_contain_url", "")
    url_matched = bool(must_url and urls_match(chunk.url or "", must_url))
    if must_url:
        return url_matched

    keywords = [normalize_answer(keyword) for keyword in (example.get("answer_keywords") or []) if keyword]
    if keywords:
        combined_content = normalize_answer(f"{chunk.title} {chunk.text}")
        if not any(keyword in combined_content for keyword in keywords):
            return False

    return True


def _gold_results_for_example(
    example: Dict[str, Any],
    chunks: Sequence[SourceChunk],
) -> List[Tuple[SourceChunk, float]]:
    gold: List[Tuple[SourceChunk, float]] = []
    must_url = example.get("must_contain_url", "")
    for chunk in chunks:
        if must_url and urls_match(chunk.url or "", must_url):
            gold.append((chunk, 1.0))
            continue
        if is_hit(chunk, example):
            gold.append((chunk, 1.0))
    return gold


def _build_reference_answer(
    example: Dict[str, Any],
    chunks: Sequence[SourceChunk],
    config: PipelineConfig,
) -> Tuple[Optional[str], str]:
    explicit_answer = (example.get("answer") or "").strip()
    if explicit_answer:
        return explicit_answer, "explicit_answer"

    gold_results = _gold_results_for_example(example, chunks)
    if not gold_results:
        return None, "no_gold_chunks"

    analysis = analyze_query_with_options(
        example.get("query", ""),
        use_query_expansion=config.use_query_expansion,
        use_adaptive_top_k=config.use_adaptive_top_k,
    )
    extracted = extract_answer_before_llm(example.get("query", ""), gold_results, analysis)
    if extracted.found and extracted.answer_text.strip():
        return extracted.answer_text.strip(), "gold_extraction"

    fallback_answer = make_answer_no_llm_struct(example.get("query", ""), list(gold_results)).answer_text.strip()
    if fallback_answer and fallback_answer != NO_KB_PHRASE:
        return fallback_answer, "gold_summary"

    return None, "no_reference_answer"


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
    ap_sum = 0.0
    ndcg_sum = 0.0
    answer_metric_count = 0
    semantic_f1_sum = 0.0
    semantic_metric_count = 0
    rouge_sum = 0.0
    rouge_metric_count = 0
    bertscore_sum = 0.0
    bertscore_metric_count = 0
    invalid_role_answers = 0
    wrong_institute_answers = 0
    hallucinations = 0
    no_reference_answer_count = 0
    groundedness_sum = 0.0
    groundedness_count = 0
    source_confidence_sum = 0.0
    source_confidence_count = 0
    trusted_source_sum = 0.0
    trusted_source_count = 0
    citation_correctness_sum = 0.0
    citation_correctness_count = 0
    latency_sum_ms = 0.0
    rerank_sum_ms = 0.0
    extraction_sum_ms = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_estimated_cost = 0.0
    cost_count = 0

    answer_config = config.to_answer_query_config(llm)

    for example in eval_set:
        query = example.get("query", "")
        started_at = time.perf_counter()
        answer, bundle = answer_query_with_config(
            query=query,
            index=index,
            chunks=chunks,
            config=answer_config,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        latency_sum_ms += latency_ms
        rerank_sum_ms += float(bundle.timings.get("rerank_ms", 0.0))
        extraction_sum_ms += float(bundle.timings.get("extraction_ms", 0.0))

        results = bundle.results
        context_results = bundle.context_results or bundle.results
        hit_rank = None
        relevant_count = 0
        gold_results = _gold_results_for_example(example, chunks)
        gold_total = len(gold_results)
        for rank, (chunk, _) in enumerate(results, start=1):
            if is_hit(chunk, example):
                relevant_count += 1
                if hit_rank is None:
                    hit_rank = rank
        precision_at_k = relevant_count / config.top_k if config.top_k else 0.0
        precision_sum += precision_at_k
        ap = average_precision(results, example, top_k=config.top_k, gold_total=gold_total)
        ap_sum += ap
        ndcg = ndcg_at_k(results, example, top_k=config.top_k, gold_total=gold_total)
        ndcg_sum += ndcg

        if hit_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / hit_rank
            if hit_rank == 1:
                hits_at_1 += 1
            if hit_rank <= 3:
                hits_at_3 += 1
            if hit_rank <= 5:
                hits_at_5 += 1

        reference_answer, reference_mode = _build_reference_answer(example, chunks, config)
        exact_match, f1 = compute_exact_match_and_f1(answer.answer_text, reference_answer)
        if exact_match is not None and f1 is not None:
            exact_sum += exact_match
            f1_sum += f1
            answer_metric_count += 1
        else:
            no_reference_answer_count += 1

        semantic_f1 = semantic_similarity(answer.answer_text, reference_answer, model_name=config.embed_model)
        if semantic_f1 is not None:
            semantic_f1_sum += semantic_f1
            semantic_metric_count += 1

        rouge_l = rouge_l_score(answer.answer_text, reference_answer)
        if rouge_l is not None:
            rouge_sum += rouge_l
            rouge_metric_count += 1

        bert_score_f1_value = bertscore_f1(answer.answer_text, reference_answer)
        if bert_score_f1_value is not None:
            bertscore_sum += bert_score_f1_value
            bertscore_metric_count += 1

        supporting_chunks = [
            context_results[idx - 1][0]
            for idx in answer.used_sources
            if 1 <= idx <= len(context_results)
        ] or [chunk for chunk, _ in context_results[: max(1, min(2, len(context_results)))]]

        grounding = groundedness_score(answer.answer_text, supporting_chunks)
        if grounding is not None:
            groundedness_sum += grounding
            groundedness_count += 1

        source_conf = mean_source_confidence(supporting_chunks)
        if source_conf is not None:
            source_confidence_sum += source_conf
            source_confidence_count += 1

        trusted_ratio = trusted_source_ratio(supporting_chunks)
        if trusted_ratio is not None:
            trusted_source_sum += trusted_ratio
            trusted_source_count += 1

        citation_score = citation_correctness(answer, context_results)
        if citation_score is not None:
            citation_correctness_sum += citation_score
            citation_correctness_count += 1

        prompt_tokens, completion_tokens = estimate_token_usage(query, answer.answer_text, supporting_chunks)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        estimated_cost = estimate_cost_usd(llm if config.use_generation else None, prompt_tokens, completion_tokens)
        if estimated_cost is not None:
            total_estimated_cost += estimated_cost
            cost_count += 1

        pred_norm = normalize_answer(answer.answer_text)
        if "заступник" in pred_norm or "адміністратор" in pred_norm:
            invalid_role_answers += 1
        if example.get("must_contain_url") and results and not urls_match(results[0][0].url or "", str(example.get("must_contain_url") or "")):
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
                "average_precision": ap,
                "ndcg_at_k": ndcg,
                "answer_text": answer.answer_text,
                "reference_answer": reference_answer,
                "reference_mode": reference_mode,
                "answer_metric_eligible": exact_match is not None,
                "confidence": answer.confidence,
                "exact_match": exact_match,
                "f1": f1,
                "semantic_f1": semantic_f1,
                "rouge_l": rouge_l,
                "bertscore_f1": bert_score_f1_value,
                "grounding_score": grounding,
                "source_confidence": source_conf,
                "trusted_source_ratio": trusted_ratio,
                "citation_correctness": citation_score,
                "latency_ms": round(latency_ms, 3),
                "rerank_ms": float(bundle.timings.get("rerank_ms", 0.0)),
                "extraction_ms": float(bundle.timings.get("extraction_ms", 0.0)),
                "retrieval_ms": float(bundle.timings.get("total_retrieval_ms", 0.0)),
                "prompt_tokens_est": prompt_tokens,
                "completion_tokens_est": completion_tokens,
                "estimated_cost_usd": estimated_cost,
                "question_type": bundle.analysis.question_type,
                "intent": bundle.analysis.intent,
                "entity_scope": bundle.analysis.entity_scope,
                "expanded_query": bundle.analysis.expanded_query,
                "used_sources": ",".join(str(src) for src in answer.used_sources),
                "entity_candidates": len(bundle.entity_candidates),
                "final_decision": bundle.final_decision,
                "warnings": " | ".join(answer.warnings or []),
            }
        )
        if exact_match != 1.0:
            error_rows.append(
                {
                    "experiment": config.name,
                    "query": query,
                    "expected_answer": reference_answer,
                    "reference_mode": reference_mode,
                    "predicted_answer": answer.answer_text,
                    "intent": bundle.analysis.intent,
                    "entity_scope": bundle.analysis.entity_scope,
                    "hit_at_1": bool(hit_rank == 1),
                    "hit_rank": hit_rank,
                    "top1_url": results[0][0].url if results else None,
                    "warnings": list(answer.warnings or []),
                    "final_decision": bundle.final_decision,
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
        "map_at_k": ap_sum / n if n else 0.0,
        "ndcg_at_k": ndcg_sum / n if n else 0.0,
        "exact_match": exact_sum / answer_metric_count if answer_metric_count else None,
        "f1": f1_sum / answer_metric_count if answer_metric_count else None,
        "semantic_f1": semantic_f1_sum / semantic_metric_count if semantic_metric_count else None,
        "rouge_l": rouge_sum / rouge_metric_count if rouge_metric_count else None,
        "bertscore_f1": bertscore_sum / bertscore_metric_count if bertscore_metric_count else None,
        "answer_metric_count": answer_metric_count,
        "answer_metric_skipped": no_reference_answer_count,
        "grounding_score": groundedness_sum / groundedness_count if groundedness_count else None,
        "source_confidence": source_confidence_sum / source_confidence_count if source_confidence_count else None,
        "trusted_source_ratio": trusted_source_sum / trusted_source_count if trusted_source_count else None,
        "citation_correctness": citation_correctness_sum / citation_correctness_count if citation_correctness_count else None,
        "avg_latency_ms": latency_sum_ms / n if n else 0.0,
        "avg_rerank_ms": rerank_sum_ms / n if n else 0.0,
        "avg_extraction_ms": extraction_sum_ms / n if n else 0.0,
        "prompt_tokens_est": total_prompt_tokens,
        "completion_tokens_est": total_completion_tokens,
        "estimated_cost_usd": total_estimated_cost if cost_count else None,
        "top1_score_mean": float(details["top1_score"].dropna().mean()) if not details["top1_score"].dropna().empty else None,
        "error_count": len(error_rows),
        "invalid_role_answers": invalid_role_answers,
        "wrong_institute_answers": wrong_institute_answers,
        "hallucinations": hallucinations,
    }
    details.attrs["errors"] = error_rows
    return metrics, details
