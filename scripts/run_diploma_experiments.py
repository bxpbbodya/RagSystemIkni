from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import CONFIG
from core.evaluation import PipelineConfig, load_eval_set, run_pipeline_evaluation
from core.index import build_faiss_index, load_chunks_from_jsonl, load_faiss_index
from core.llm import LLMSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diploma RAG experiments")
    parser.add_argument("--eval-set", default="eval_set.jsonl")
    parser.add_argument("--report-dir", default="report/diploma")
    parser.add_argument("--index-cache-dir", default="data/model_indexes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--embed-model", action="append", dest="embed_models")
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--compare-llms", action="store_true")
    return parser.parse_args()


def slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def ensure_index_for_model(chunks, embed_model: str, cache_dir: Path, *, rebuild: bool = False):
    slug = slugify(embed_model)
    index_path = cache_dir / f"{slug}.faiss"
    meta_path = cache_dir / f"{slug}.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if rebuild or not index_path.exists() or not meta_path.exists():
        print(f"[INDEX] Building index for {embed_model}")
        build_faiss_index(
            chunks=chunks,
            embed_model_name=embed_model,
            index_path=index_path,
            meta_path=meta_path,
        )
    return load_faiss_index(index_path, meta_path)


def build_llm_settings() -> List[Tuple[str, LLMSettings]]:
    configs: List[Tuple[str, LLMSettings]] = []

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        configs.append(
            (
                "gpt-4o-mini",
                LLMSettings(
                    enabled=True,
                    provider="openai",
                    model="gpt-4o-mini",
                    api_key=openai_key,
                    temperature=0.0,
                    max_tokens=180,
                ),
            )
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        configs.append(
            (
                "mistral-small",
                LLMSettings(
                    enabled=True,
                    provider="openrouter",
                    model="mistralai/mistral-small-3.1-24b-instruct",
                    api_key=openrouter_key,
                    temperature=0.0,
                    max_tokens=180,
                ),
            )
        )
        configs.append(
            (
                "llama-3.1-70b",
                LLMSettings(
                    enabled=True,
                    provider="openrouter",
                    model="meta-llama/llama-3.1-70b-instruct",
                    api_key=openrouter_key,
                    temperature=0.0,
                    max_tokens=180,
                ),
            )
        )
    return configs


def build_experiments(embed_models: List[str], top_k: int, min_score: float, reranker_model: str) -> List[PipelineConfig]:
    experiments: List[PipelineConfig] = []
    for embed_model in embed_models:
        suffix = slugify(embed_model)
        experiments.extend(
            [
                PipelineConfig(
                    name=f"baseline_embeddings_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=False,
                    use_post_boosts=False,
                    use_reranker=False,
                    use_query_expansion=False,
                    use_adaptive_top_k=False,
                    use_generation=False,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"hybrid_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=False,
                    use_reranker=False,
                    use_query_expansion=False,
                    use_adaptive_top_k=False,
                    use_generation=False,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"hybrid_reranker_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=False,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=False,
                    use_adaptive_top_k=False,
                    use_generation=False,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"hybrid_reranker_queryexp_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=True,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_generation=False,
                    use_institute_filter=True,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"hybrid_reranker_queryexp_no_institute_filter_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=True,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_generation=False,
                    use_institute_filter=False,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"full_pipeline_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=True,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_generation=True,
                    use_extraction=True,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"full_pipeline_no_extraction_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    use_hybrid=True,
                    use_post_boosts=True,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_generation=True,
                    use_extraction=False,
                    keyword_filter=False,
                    min_score=min_score,
                ),
                PipelineConfig(
                    name=f"local_only_pipeline_{suffix}",
                    embed_model=embed_model,
                    top_k=top_k,
                    allowed_types=("local",),
                    use_hybrid=True,
                    use_post_boosts=True,
                    use_reranker=True,
                    reranker_model=reranker_model,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_generation=True,
                    keyword_filter=False,
                    min_score=min_score,
                ),
            ]
        )
    return experiments


def save_outputs(report_dir: Path, summary: pd.DataFrame, details: pd.DataFrame) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "diploma_summary.csv", index=False, encoding="utf-8")
    details.to_csv(report_dir / "diploma_details.csv", index=False, encoding="utf-8")
    if "exact_match" in details.columns:
        errors = details[details["exact_match"] < 1.0]
        errors.to_csv(report_dir / "diploma_errors.csv", index=False, encoding="utf-8")
    best = summary.sort_values(["hit_at_1", "mrr_at_k", "exact_match", "f1"], ascending=False).iloc[0].to_dict()
    (report_dir / "diploma_best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_set = load_eval_set(Path(args.eval_set))
    if not eval_set:
        raise SystemExit(f"Evaluation set not found or empty: {args.eval_set}")

    source_path = CONFIG.local_cache_path if CONFIG.local_cache_path.exists() else CONFIG.faiss_meta_path
    chunks = load_chunks_from_jsonl(source_path)
    if not chunks:
        raise SystemExit("No chunks found. Run sync/index build first.")

    embed_models = args.embed_models or [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
    ]
    llm_variants = build_llm_settings() if args.compare_llms else []

    experiments = build_experiments(embed_models, args.top_k, args.min_score, args.reranker_model)

    model_indexes: Dict[str, Tuple[Any, List[Any]]] = {}
    summary_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []

    for experiment in experiments:
        if experiment.embed_model not in model_indexes:
            model_indexes[experiment.embed_model] = ensure_index_for_model(
                chunks,
                experiment.embed_model,
                Path(args.index_cache_dir),
                rebuild=args.rebuild_indexes,
            )

        index, model_chunks = model_indexes[experiment.embed_model]
        llm_settings = None
        llm_variants_to_run = [(None, None)]
        if experiment.use_generation and llm_variants:
            llm_variants_to_run = llm_variants

        for llm_name, llm_settings in llm_variants_to_run:
            experiment_name = experiment.name if not llm_name else f"{experiment.name}_{slugify(llm_name)}"
            config = PipelineConfig(**{**experiment.__dict__, "name": experiment_name})
            print(f"[RUN] {config.name}")
            metrics, details = run_pipeline_evaluation(
                eval_set=eval_set,
                index=index,
                chunks=model_chunks,
                config=config,
                llm=llm_settings,
            )
            if llm_name:
                metrics["llm_model"] = llm_name
                details["llm_model"] = llm_name
            else:
                metrics["llm_model"] = None
                details["llm_model"] = None
            summary_rows.append(metrics)
            detail_frames.append(details)

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["experiment"].str.startswith("baseline_embeddings")]
    if baseline.empty:
        baseline_recall = 0.0
    else:
        baseline_recall = float(baseline.iloc[0]["recall_at_k"])
    summary["delta_recall_vs_baseline"] = summary["recall_at_k"] - baseline_recall
    summary = summary.sort_values(["hit_at_1", "mrr_at_k", "exact_match", "f1"], ascending=False)

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    save_outputs(Path(args.report_dir), summary, details)

    display_columns = ["experiment", "embed_model", "recall_at_k", "mrr_at_k", "hit_at_1", "exact_match", "f1", "delta_recall_vs_baseline"]
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
