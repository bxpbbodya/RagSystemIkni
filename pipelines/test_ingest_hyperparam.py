# hyperparam_search.py

import asyncio
import itertools
import csv
from pathlib import Path
import json
from typing import List, Dict, Tuple

from pipelines.ingest_lpnu import ingest_lpnu_async
from core.index import (
    build_faiss_index,
    load_chunks_from_jsonl,
    search_index,
    SourceChunk
)

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_index_with_queries(
    index,
    chunks: List[SourceChunk],
    queries_file: str,
    embed_model_name: str,
    top_k: int = 5,
    min_score: float = 0.2,
    keyword_filter: bool = True,
) -> Dict[str, float]:

    with open(queries_file, "r", encoding="utf-8") as f:
        eval_queries = [json.loads(line) for line in f]

    total = len(eval_queries)

    hit1 = hit3 = hit5 = 0
    recall = mrr = avgp = 0

    for q in eval_queries:
        query = q["query"]
        keywords = set(k.lower() for k in q.get("answer_keywords", []))
        must_url = q.get("must_contain_url", "")

        results = search_index(
            query=query,
            index=index,
            chunks=chunks,
            embed_model_name=embed_model_name,
            top_k=top_k,
            min_score=min_score,
            keyword_filter=keyword_filter,
        )

        hits = []
        for rank, (ch, _) in enumerate(results[:top_k]):
            text = (ch.text or "").lower()
            url_ok = must_url in (ch.url or "")
            kw_ok = all(k in text for k in keywords)

            if url_ok and kw_ok:
                hits.append(rank + 1)

        if hits:
            first = hits[0]

            if first == 1:
                hit1 += 1
            if first <= 3:
                hit3 += 1
            if first <= 5:
                hit5 += 1

            recall += 1
            mrr += 1 / first
            avgp += 1 / len(hits)

    return {
        "recall@k": recall / total,
        "MRR@k": mrr / total,
        "avg_precision@k": avgp / total,
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total,
    }


# -----------------------------
# Hyperparam search
# -----------------------------
async def run_hyperparam_search():

    models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "LaBSE"
    ]

    chunk_sizes = [150, 200, 300]
    overlaps = [50, 70]
    min_scores = [0.05, 0.1, 0.2]
    keyword_filters = [True, False]

    # ❗ прибрав crawler_mode і reranker (вони зараз нічого не роблять)
    # якщо хочеш — потім нормально додамо

    results_file = Path("data/hyperparam_results.csv")
    results_file.parent.mkdir(exist_ok=True)

    with results_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "chunk_size", "overlap", "min_score", "keyword_filter",
            "recall@k", "MRR@k", "avg_precision@k", "hit@1", "hit@3", "hit@5"
        ])

    # 🔥 ГРУПУЄМО ПО КЕШУ (щоб не кравлити кожен раз)
    cache_map = {}

    for chunk_size, overlap in itertools.product(chunk_sizes, overlaps):
        cache_file = f"data/cache_{chunk_size}_{overlap}.jsonl"

        print(f"\n[CRAWL] chunk={chunk_size}, overlap={overlap}")

        await ingest_lpnu_async(
            cache_path=cache_file,
            chunk_size=chunk_size,
            overlap=overlap
        )

        chunks = load_chunks_from_jsonl(Path(cache_file))
        cache_map[(chunk_size, overlap)] = chunks

    # 🔥 ОСНОВНИЙ GRID SEARCH
    for model, chunk_size, overlap, min_score, kw_filter in itertools.product(
        models, chunk_sizes, overlaps, min_scores, keyword_filters
    ):

        print(
            f"[TEST] {model} | chunk={chunk_size} | overlap={overlap} | "
            f"min_score={min_score} | kw={kw_filter}"
        )

        chunks = cache_map[(chunk_size, overlap)]

        if not chunks:
            print("[WARN] empty chunks, skip")
            continue

        # Build index
        index, _ = build_faiss_index(
            chunks=chunks,
            embed_model_name=model,
            index_path=Path("data/tmp.faiss"),
            meta_path=Path("data/tmp_meta.jsonl")
        )

        # Evaluate
        metrics = evaluate_index_with_queries(
            index=index,
            chunks=chunks,
            queries_file="eval_set.jsonl",
            embed_model_name=model,
            min_score=min_score,
            keyword_filter=kw_filter,
        )

        # Save
        with results_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                model, chunk_size, overlap, min_score, kw_filter,
                metrics["recall@k"],
                metrics["MRR@k"],
                metrics["avg_precision@k"],
                metrics["hit@1"],
                metrics["hit@3"],
                metrics["hit@5"]
            ])

        print(f"[RESULT] {metrics}")


if __name__ == "__main__":
    asyncio.run(run_hyperparam_search())