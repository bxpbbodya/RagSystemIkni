from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .config import CONFIG
from .evaluation import PipelineConfig, load_eval_set, run_pipeline_evaluation
from .index import load_faiss_index
from .llm import LLMSettings


def build_benchmark_profiles(
    *,
    embed_model: str,
    reranker_model: Optional[str],
    top_k: int,
    min_score: float,
    allowed_types: Optional[Sequence[str]] = None,
) -> List[PipelineConfig]:
    allowed = tuple(allowed_types) if allowed_types else None
    return [
        PipelineConfig(
            name="baseline_v1_vector_only",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=False,
            use_post_boosts=False,
            use_reranker=False,
            use_query_expansion=False,
            use_adaptive_top_k=False,
            use_generation=False,
            use_extraction=False,
            use_institute_filter=False,
            use_rules=False,
            keyword_filter=False,
            min_score=min_score,
        ),
        PipelineConfig(
            name="baseline_v2_vector_plus_summary",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=False,
            use_post_boosts=False,
            use_reranker=False,
            use_query_expansion=False,
            use_adaptive_top_k=False,
            use_generation=False,
            use_extraction=False,
            use_institute_filter=False,
            use_rules=False,
            keyword_filter=False,
            min_score=min_score,
        ),
        PipelineConfig(
            name="hybrid_retrieval",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=True,
            use_post_boosts=False,
            use_reranker=False,
            use_query_expansion=False,
            use_adaptive_top_k=False,
            use_generation=False,
            use_extraction=False,
            use_institute_filter=False,
            use_rules=False,
            keyword_filter=False,
            min_score=min_score,
        ),
        PipelineConfig(
            name="hybrid_plus_reranker",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=True,
            use_post_boosts=False,
            use_reranker=bool(reranker_model),
            reranker_model=reranker_model,
            use_query_expansion=False,
            use_adaptive_top_k=False,
            use_generation=False,
            use_extraction=False,
            use_institute_filter=False,
            use_rules=False,
            keyword_filter=False,
            min_score=min_score,
        ),
        PipelineConfig(
            name="adaptive_hybrid_queryexp",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=True,
            use_post_boosts=True,
            use_reranker=bool(reranker_model),
            reranker_model=reranker_model,
            use_query_expansion=True,
            use_adaptive_top_k=True,
            use_generation=False,
            use_extraction=True,
            use_institute_filter=True,
            use_rules=True,
            keyword_filter=False,
            min_score=min_score,
        ),
        PipelineConfig(
            name="production_full_pipeline",
            embed_model=embed_model,
            top_k=top_k,
            allowed_types=allowed,
            use_hybrid=True,
            use_post_boosts=True,
            use_reranker=bool(reranker_model),
            reranker_model=reranker_model,
            use_query_expansion=True,
            use_adaptive_top_k=True,
            use_generation=True,
            use_extraction=True,
            use_institute_filter=True,
            use_rules=True,
            keyword_filter=False,
            min_score=min_score,
        ),
    ]


def run_benchmark_suite(
    *,
    eval_set_path: str | Path = "eval_set.jsonl",
    report_root: str | Path = "report/benchmarks",
    embed_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.2,
    llm: Optional[LLMSettings] = None,
    allowed_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    eval_set_path = Path(eval_set_path)
    report_root = Path(report_root)
    eval_set = load_eval_set(eval_set_path)
    if not eval_set:
        raise ValueError(f"Evaluation set not found or empty: {eval_set_path}")

    index, chunks = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
    profiles = build_benchmark_profiles(
        embed_model=embed_model or CONFIG.embed_model_name,
        reranker_model=reranker_model or CONFIG.reranker_model_name,
        top_k=top_k,
        min_score=min_score,
        allowed_types=allowed_types,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_set_path": str(eval_set_path),
        "profiles": [asdict(profile) for profile in profiles],
    }

    for profile in profiles:
        effective_llm = llm if (profile.use_generation and llm and llm.enabled) else None
        effective_profile = PipelineConfig(**{**asdict(profile), "use_generation": bool(effective_llm and profile.use_generation)})
        metrics, details = run_pipeline_evaluation(
            eval_set=eval_set,
            index=index,
            chunks=chunks,
            config=effective_profile,
            llm=effective_llm,
        )
        summary_rows.append(metrics)
        detail_frames.append(details)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    baseline_v1 = summary.loc[summary["experiment"] == "baseline_v1_vector_only"]
    baseline_v2 = summary.loc[summary["experiment"] == "baseline_v2_vector_plus_summary"]
    base1 = baseline_v1.iloc[0].to_dict() if not baseline_v1.empty else {}
    base2 = baseline_v2.iloc[0].to_dict() if not baseline_v2.empty else {}
    for metric_name in ["recall_at_k", "mrr_at_k", "hit_at_1", "exact_match", "f1", "ndcg_at_k"]:
        if metric_name in summary.columns:
            summary[f"gain_vs_v1_{metric_name}"] = summary[metric_name] - float(base1.get(metric_name) or 0.0)
            summary[f"gain_vs_v2_{metric_name}"] = summary[metric_name] - float(base2.get(metric_name) or 0.0)

    summary = summary.sort_values(["hit_at_1", "mrr_at_k", "exact_match", "f1"], ascending=False)
    summary.to_csv(run_dir / "summary.csv", index=False, encoding="utf-8")
    details.to_csv(run_dir / "details.csv", index=False, encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "details": details,
    }


def load_benchmark_history(report_root: str | Path = "report/benchmarks") -> pd.DataFrame:
    report_root = Path(report_root)
    rows: List[Dict[str, Any]] = []
    if not report_root.exists():
        return pd.DataFrame()

    for run_dir in sorted(report_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.csv"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.exists():
            continue
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            continue
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        top_row = summary.iloc[0].to_dict() if not summary.empty else {}
        rows.append(
            {
                "run_id": run_dir.name,
                "created_at": manifest.get("created_at"),
                "profiles": len(manifest.get("profiles", [])),
                "best_experiment": top_row.get("experiment"),
                "best_hit_at_1": top_row.get("hit_at_1"),
                "best_mrr_at_k": top_row.get("mrr_at_k"),
                "best_exact_match": top_row.get("exact_match"),
                "best_f1": top_row.get("f1"),
                "path": str(run_dir),
            }
        )
    return pd.DataFrame(rows)
