from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


QUESTION_PREFIXES = (
    "Підкажи, будь ласка,",
    "Скажи, будь ласка,",
    "Можеш уточнити,",
    "Потрібно дізнатися,",
)


def load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def save_jsonl_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def typo_variants(query: str) -> List[str]:
    query = (query or "").strip()
    if len(query) < 8:
        return []
    variants = []
    variants.append(query.replace("і", "и", 1))
    variants.append(query.replace("?", ""))
    variants.append(query.replace(" ", "  ", 1))
    variants.append(query[:-1] if query.endswith("?") else query + " ?")
    return list(dict.fromkeys(v for v in variants if v and v != query))


def paraphrase_variants(query: str) -> List[str]:
    query = (query or "").strip()
    if not query:
        return []
    variants = [f"{prefix} {query[0].lower() + query[1:]}" for prefix in QUESTION_PREFIXES if len(query) > 1]
    if query.lower().startswith("хто "):
        variants.append(query.replace("Хто", "Назви", 1))
    if query.lower().startswith("як "):
        variants.append(query.replace("Як", "Яким чином", 1))
    if query.lower().startswith("де "):
        variants.append(query.replace("Де", "В якому місці", 1))
    return list(dict.fromkeys(v for v in variants if v and v != query))


def noisy_variants(query: str) -> List[str]:
    query = (query or "").strip()
    if not query:
        return []
    return list(
        dict.fromkeys(
            [
                f"Терміново: {query}",
                f"{query} зараз",
                f"{query} для студента",
            ]
        )
    )


def expand_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_query = (example.get("query") or "").strip()
    if not base_query:
        return []

    rows = [{**example, "variant_type": "base"}]
    for variant in paraphrase_variants(base_query):
        rows.append({**example, "query": variant, "variant_type": "paraphrase"})
    for variant in typo_variants(base_query):
        rows.append({**example, "query": variant, "variant_type": "typo"})
    for variant in noisy_variants(base_query):
        rows.append({**example, "query": variant, "variant_type": "noisy"})
    return rows


def feedback_to_examples(feedback_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in feedback_rows:
        if int(row.get("rating", 0)) >= 0:
            continue
        query = (row.get("query") or "").strip()
        if not query:
            continue
        sources = row.get("retrieval_sources") or []
        top_url = None
        if sources:
            top_url = sources[0].get("url")
        examples.append(
            {
                "query": query,
                "must_contain_url": top_url,
                "answer": "",
                "answer_keywords": [],
                "category": "feedback_hard_case",
                "variant_type": "feedback",
            }
        )
    return examples


def expand_eval_dataset(
    *,
    source_path: str | Path = "eval_set.jsonl",
    output_path: str | Path = "report/eval_expanded.jsonl",
    feedback_path: str | Path = "data/feedback.jsonl",
) -> Dict[str, Any]:
    base_rows = load_jsonl_rows(source_path)
    feedback_rows = load_jsonl_rows(feedback_path)

    expanded: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for row in base_rows:
        for variant in expand_example(row):
            key = (
                (variant.get("query") or "").strip().lower(),
                str(variant.get("must_contain_url") or "").strip().lower(),
                str(variant.get("category") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            expanded.append(variant)

    for row in feedback_to_examples(feedback_rows):
        key = (
            (row.get("query") or "").strip().lower(),
            str(row.get("must_contain_url") or "").strip().lower(),
            str(row.get("category") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        expanded.append(row)

    save_jsonl_rows(output_path, expanded)
    return {
        "ok": True,
        "base_examples": len(base_rows),
        "feedback_cases": len(feedback_to_examples(feedback_rows)),
        "expanded_examples": len(expanded),
        "output_path": str(output_path),
    }
