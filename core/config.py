# core/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class AppConfig:
    project_name: str = "IKNI Data Analysis Assistant (RAG MVP)"
    local_cache_path: Path = BASE_DIR / "data" / "local_cache.jsonl"
    faiss_index_path: Path = BASE_DIR / "data" / "index.faiss"
    faiss_meta_path: Path = BASE_DIR / "data" / "index_meta.jsonl"
    report_dir: Path = BASE_DIR / "report"
    benchmark_dir: Path = BASE_DIR / "report" / "benchmarks"
    expanded_eval_path: Path = BASE_DIR / "report" / "eval_expanded.jsonl"
    self_improve_report_path: Path = BASE_DIR / "report" / "self_improve_report.json"

    # Embeddings
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 5

    # ✅ Reranker
    use_reranker: bool = False
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_n: int = 30

    internal_k_min: int = 30
    internal_k_multiplier: int = 8


CONFIG = AppConfig()
