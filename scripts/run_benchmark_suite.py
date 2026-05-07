from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.benchmarking import run_benchmark_suite


def main() -> None:
    result = run_benchmark_suite()
    print(json.dumps({"run_id": result["run_id"], "run_dir": result["run_dir"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
