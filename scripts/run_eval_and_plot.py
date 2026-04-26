from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import CONFIG
from core.evaluation import PipelineConfig, load_eval_set, run_pipeline_evaluation
from core.index import load_faiss_index


def main() -> None:
    eval_set = load_eval_set(Path("eval_set.jsonl"))
    if not eval_set:
        raise SystemExit("eval_set.jsonl not found or empty.")

    index, chunks = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
    config = PipelineConfig(
        name="single_eval_pipeline",
        embed_model=CONFIG.embed_model_name,
        top_k=5,
        use_hybrid=True,
        use_post_boosts=True,
        use_reranker=False,
        use_query_expansion=True,
        use_adaptive_top_k=True,
        use_generation=False,
        use_extraction=True,
        use_rules=True,
        keyword_filter=False,
        min_score=0.2,
    )

    metrics, details = run_pipeline_evaluation(
        eval_set=eval_set,
        index=index,
        chunks=chunks,
        config=config,
        llm=None,
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
