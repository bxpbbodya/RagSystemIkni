from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import CONFIG
from core.kb_validation import validate_local_knowledge_base


def main() -> None:
    report = validate_local_knowledge_base(CONFIG.local_cache_path, eval_set_path="eval_set.jsonl")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    except Exception:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
