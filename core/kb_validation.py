from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .index import load_chunks_from_jsonl


def validate_local_knowledge_base(
    cache_path: str | Path,
    *,
    eval_set_path: str | Path = "eval_set.jsonl",
) -> Dict[str, Any]:
    cache_path = Path(cache_path)
    eval_set_path = Path(eval_set_path)
    chunks = load_chunks_from_jsonl(cache_path)

    source_counts = Counter((chunk.source_type or "unknown").lower() for chunk in chunks)
    url_counts = Counter((chunk.url or "").rstrip("/") for chunk in chunks if chunk.url)
    duplicate_urls = {url: count for url, count in url_counts.items() if count > 3}

    target_urls: List[str] = []
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
                url = (obj.get("must_contain_url") or "").rstrip("/")
                if url and url not in target_urls:
                    target_urls.append(url)

    cached_urls = set(url_counts.keys())
    covered = [url for url in target_urls if url in cached_urls]
    missing = [url for url in target_urls if url not in cached_urls]

    return {
        "total_chunks": len(chunks),
        "source_counts": dict(source_counts),
        "unique_urls": len(url_counts),
        "duplicate_urls": duplicate_urls,
        "eval_target_total": len(target_urls),
        "eval_target_covered": len(covered),
        "eval_target_missing": missing,
    }
