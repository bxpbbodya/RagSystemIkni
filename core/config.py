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
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 5

CONFIG = AppConfig()
