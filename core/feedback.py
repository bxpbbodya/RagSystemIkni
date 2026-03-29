# core/feedback.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

DEFAULT_PATH = Path("data/feedback.jsonl")

def save_feedback(
    *,
    query: str,
    verdict: str,  # good | bad_answer | bad_retrieval | no_answer
    comment: str = "",
    top_urls: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    path: Path = DEFAULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "query": query.strip(),
        "verdict": verdict,
        "comment": comment.strip(),
        "top_urls": top_urls or [],
        "meta": meta or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_feedback(path: Path = DEFAULT_PATH, limit: int = 2000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]
