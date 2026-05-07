from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import CONFIG
from .eval_dataset import expand_eval_dataset, load_jsonl_rows, save_jsonl_rows
from .feedback import load_feedback
from .kb_validation import validate_local_knowledge_base
from pipelines.sync_all import rebuild_index


def collect_hard_cases(
    *,
    feedback_path: str | Path = "data/feedback.jsonl",
    eval_errors_path: str | Path = "report/eval_errors.jsonl",
    output_path: str | Path = "data/self_improve_pool.jsonl",
) -> Dict[str, Any]:
    feedback_rows = load_feedback(Path(feedback_path), limit=5000)
    eval_errors = load_jsonl_rows(eval_errors_path)

    hard_cases: List[Dict[str, Any]] = []
    for row in feedback_rows:
        if int(row.get("rating", 0)) >= 0:
            continue
        hard_cases.append(
            {
                "source": "feedback",
                "query": row.get("query"),
                "comment": row.get("comment"),
                "ts": row.get("ts"),
                "meta": row,
            }
        )

    for row in eval_errors:
        hard_cases.append(
            {
                "source": "evaluation",
                "query": row.get("query"),
                "comment": row.get("predicted_answer"),
                "ts": datetime.now().isoformat(timespec="seconds"),
                "meta": row,
            }
        )

    save_jsonl_rows(output_path, hard_cases)
    return {
        "ok": True,
        "hard_cases": len(hard_cases),
        "output_path": str(output_path),
    }


def run_self_improve(
    *,
    reindex: bool = True,
    expanded_eval_output: str | Path = "report/eval_expanded.jsonl",
    report_path: str | Path = "report/self_improve_report.json",
) -> Dict[str, Any]:
    hard_case_report = collect_hard_cases()
    expanded_eval_report = expand_eval_dataset(output_path=expanded_eval_output)

    reindex_report: Dict[str, Any] = {"ok": False, "skipped": True}
    if reindex:
        reindex_report = rebuild_index()

    kb_report = validate_local_knowledge_base(CONFIG.local_cache_path, eval_set_path=expanded_eval_output)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hard_case_report": hard_case_report,
        "expanded_eval_report": expanded_eval_report,
        "reindex_report": reindex_report,
        "kb_report": kb_report,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
