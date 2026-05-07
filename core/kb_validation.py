from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .evaluation import urls_match
from .index import load_chunks_from_jsonl
from .query_pipeline import source_trust_score


NAME_RE = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ'’\-]+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ'’\-]+){1,2}")


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _freshness_buckets(chunks) -> Dict[str, int]:
    buckets = Counter()
    for chunk in chunks:
        date = (chunk.date or "").strip()
        if not date:
            buckets["unknown"] += 1
        elif date[:4].isdigit():
            buckets[date[:4]] += 1
        else:
            buckets["unknown"] += 1
    return dict(buckets)


def _detect_conflicts(chunks) -> List[Dict[str, Any]]:
    grouped: Dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        haystack = f"{chunk.title or ''}\n{chunk.text or ''}".lower()
        if "директор" not in haystack and "ректор" not in haystack:
            continue
        names = set(NAME_RE.findall(f"{chunk.title or ''}\n{chunk.text or ''}"))
        if not names:
            continue
        grouped[_normalize_url(chunk.url or chunk.title or chunk.chunk_id)].update(names)

    conflicts = []
    for key, names in grouped.items():
        if len(names) > 1:
            conflicts.append({"target": key, "names": sorted(names)})
    return conflicts[:50]


def validate_local_knowledge_base(
    cache_path: str | Path,
    *,
    eval_set_path: str | Path = "eval_set.jsonl",
) -> Dict[str, Any]:
    cache_path = Path(cache_path)
    eval_set_path = Path(eval_set_path)
    chunks = load_chunks_from_jsonl(cache_path)

    source_counts = Counter((chunk.source_type or "unknown").lower() for chunk in chunks)
    url_counts = Counter(_normalize_url(chunk.url or "") for chunk in chunks if chunk.url)
    duplicate_urls = {url: count for url, count in url_counts.items() if count > 3}
    trust_scores = [source_trust_score(chunk) for chunk in chunks]
    word_counts = [int((chunk.extra or {}).get("word_count") or len((chunk.text or "").split())) for chunk in chunks]

    target_urls: List[str] = []
    eval_rows: List[Dict[str, Any]] = []
    if eval_set_path.exists():
        with eval_set_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eval_rows.append(obj)
                url = _normalize_url(obj.get("must_contain_url") or "")
                if url and url not in target_urls:
                    target_urls.append(url)

    cached_urls = list(url_counts.keys())
    covered = [url for url in target_urls if any(urls_match(cached, url) for cached in cached_urls)]
    missing = [url for url in target_urls if not any(urls_match(cached, url) for cached in cached_urls)]

    contradictions = _detect_conflicts(chunks)
    trust_by_source = {
        source: round(
            sum(source_trust_score(chunk) for chunk in chunks if (chunk.source_type or "").lower() == source) /
            max(1, sum(1 for chunk in chunks if (chunk.source_type or "").lower() == source)),
            4,
        )
        for source in source_counts.keys()
    }

    return {
        "total_chunks": len(chunks),
        "source_counts": dict(source_counts),
        "source_trust_mean": round(sum(trust_scores) / len(trust_scores), 4) if trust_scores else None,
        "source_trust_by_type": trust_by_source,
        "unique_urls": len(url_counts),
        "duplicate_urls": duplicate_urls,
        "duplicate_url_ratio": round(len(duplicate_urls) / len(url_counts), 4) if url_counts else 0.0,
        "avg_words_per_chunk": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0,
        "freshness_buckets": _freshness_buckets(chunks),
        "eval_target_total": len(target_urls),
        "eval_target_covered": len(covered),
        "eval_target_missing": missing,
        "coverage_ratio": round(len(covered) / len(target_urls), 4) if target_urls else None,
        "contradiction_candidates": contradictions,
    }
