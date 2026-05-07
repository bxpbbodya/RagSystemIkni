from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import CONFIG
from core.evaluation import PipelineConfig, load_eval_set, run_pipeline_evaluation
from core.index import load_faiss_index
from core.llm import LLMSettings


def build_llm_from_env() -> LLMSettings | None:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        return None
    return LLMSettings(
        enabled=True,
        provider="openai",
        model=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"),
        api_key=openai_key,
        temperature=0.0,
        max_tokens=180,
    )


def main() -> None:
    eval_set = load_eval_set(Path("eval_set.jsonl"))
    if not eval_set:
        raise SystemExit("eval_set.jsonl not found or empty.")

    index, chunks = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
    llm = build_llm_from_env()
    config = PipelineConfig(
        name="eval_production_like",
        embed_model=CONFIG.embed_model_name,
        top_k=5,
        use_hybrid=True,
        use_post_boosts=True,
        use_reranker=True,
        use_query_expansion=True,
        use_generation=bool(llm and llm.enabled),
        use_extraction=True,
        min_score=0.15,
        keyword_filter=True,
    )

    metrics, details = run_pipeline_evaluation(
        eval_set=eval_set,
        index=index,
        chunks=chunks,
        config=config,
        llm=llm,
    )

    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    details.to_csv(report_dir / "eval_results.csv", index=False, encoding="utf-8")
    errors = list(getattr(details, "attrs", {}).get("errors", []) or [])
    (report_dir / "eval_errors.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in errors),
        encoding="utf-8",
    )
    (report_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = pd.DataFrame([metrics])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
