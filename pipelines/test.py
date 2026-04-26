import json
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from core.sources import SourceChunk
from core.index import search_index


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVAL_PATH = BASE_DIR / "eval_set.jsonl"
DATA_DIR = BASE_DIR / "data"

EMBED_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

TOP_K = 5


# =========================================================
# LOAD
# =========================================================

def load_eval():
    data = []
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_index(model_name):
    safe_name = model_name.replace("/", "_")
    index_path = DATA_DIR / f"index_{safe_name}.faiss"
    meta_path = DATA_DIR / f"index_{safe_name}_meta.jsonl"

    index = faiss.read_index(str(index_path))

    chunks = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(SourceChunk(**json.loads(line)))

    return index, chunks


# =========================================================
# METRICS
# =========================================================

def recall_at_k(retrieved, keywords):
    text = " ".join([r[0].text.lower() for r in retrieved])
    return float(any(k.lower() in text for k in keywords))


def hit_at_1(retrieved, keywords):
    if not retrieved:
        return 0.0
    text = retrieved[0][0].text.lower()
    return float(any(k.lower() in text for k in keywords))


def mrr(retrieved, keywords):
    for i, (chunk, _) in enumerate(retrieved):
        if any(k.lower() in chunk.text.lower() for k in keywords):
            return 1.0 / (i + 1)
    return 0.0


# =========================================================
# BENCHMARK
# =========================================================

def run_model(eval_set, model_name):
    print(f"\n🚀 MODEL: {model_name}")

    index, chunks = load_index(model_name)

    model = SentenceTransformer(model_name)

    recall_list, mrr_list, hit_list = [], [], []

    for sample in tqdm(eval_set, desc=model_name):
        query = sample.get("query") or sample.get("question")
        keywords = sample.get("answer_keywords", [])

        vec = model.encode([query])[0]

        D, I = index.search(np.array([vec]), TOP_K)

        retrieved = [(chunks[i], D[0][j]) for j, i in enumerate(I[0]) if i < len(chunks)]

        recall_list.append(recall_at_k(retrieved, keywords))
        mrr_list.append(mrr(retrieved, keywords))
        hit_list.append(hit_at_1(retrieved, keywords))

    return {
        "model": model_name,
        "recall@5": np.mean(recall_list),
        "mrr": np.mean(mrr_list),
        "hit@1": np.mean(hit_list),
    }


# =========================================================
# MAIN
# =========================================================

def run():
    start_time = time.time()

    eval_set = load_eval()

    results = []

    for model_name in EMBED_MODELS:
        try:
            metrics = run_model(eval_set, model_name)
            results.append(metrics)
            print(metrics)
        except Exception as e:
            print(f"[ERROR] {model_name} failed:", e)

    # sort
    results.sort(key=lambda x: x["mrr"], reverse=True)

    os.makedirs("report", exist_ok=True)

    with open("report/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n🏆 FINAL RANKING:")
    for r in results:
        print(r)

    print(f"\n⏱ Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    run()