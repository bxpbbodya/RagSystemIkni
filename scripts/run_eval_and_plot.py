# pipelines/run_eval_and_plot.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from core.config import CONFIG
from core.index import load_faiss_index, search_index, build_faiss_index
from core.sources import SourceChunk
import json

# -----------------------------
# Eval set loader
# -----------------------------
def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    eval_set = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            eval_set.append(json.loads(line))
    return eval_set

# -----------------------------
# Simple hit check
# -----------------------------
def is_hit(chunk: SourceChunk, rule: dict) -> bool:
    url = (chunk.url or "").lower()
    title = (chunk.title or "").lower()
    source_type = (chunk.source_type or "").lower()
    text = (chunk.text or "").lower()

    must_url = (rule.get("must_contain_url") or "").lower()
    must_type = (rule.get("must_contain_type") or "").lower()
    must_text = (rule.get("must_contain_text") or "").lower()

    if must_url and must_url not in url:
        return False
    if must_type and must_type != source_type:
        return False
    if must_text and must_text not in text and must_text not in title:
        return False
    return True

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(
    eval_set: List[Dict[str, Any]],
    embed_model_name: str,
    top_ks: List[int] = [1,3,5,10,15],
    reranker: Optional[CrossEncoder] = None,
    min_score: float = 0.2
) -> Dict[str, Any]:
    index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)

    rows = []
    recall_metrics = {k:0 for k in top_ks}
    rr_metrics = {k:0.0 for k in top_ks}

    for ex in tqdm(eval_set, desc=f"Evaluating {embed_model_name}"):
        query = ex["query"]
        results = search_index(
            query=query,
            index=index,
            chunks=meta,
            embed_model_name=embed_model_name,
            top_k=max(top_ks),
            min_score=min_score
        )

        # Reranking with cross-encoder
        if reranker and results:
            pairs = [(query, chunk.text) for chunk, _ in results]
            scores = reranker.predict(pairs)
            results = sorted(zip([c for c,_ in results], scores), key=lambda x: x[1], reverse=True)

        hit_rank: Optional[int] = None
        for i, (chunk, score) in enumerate(results, start=1):
            if is_hit(chunk, ex):
                hit_rank = i
                break

        for k in top_ks:
            if hit_rank is not None and hit_rank <= k:
                recall_metrics[k] += 1
                rr_metrics[k] += 1.0 / hit_rank

        rows.append({
            "query": query,
            "hit_rank": hit_rank,
            "top1_score": float(results[0][1]) if results else None,
            "top1_url": results[0][0].url if results else None,
        })

    n = len(eval_set)
    recall_at_k = {f"recall_at_{k}": recall_metrics[k]/n for k in top_ks}
    mrr_at_k = {f"mrr_at_{k}": rr_metrics[k]/n for k in top_ks}

    return {
        "n": n,
        "top_ks": top_ks,
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "rows": rows,
    }

# -----------------------------
# Plotting
# -----------------------------
def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hit ratio pie
    plt.figure()
    hit_counts = df["hit_rank"].notna().value_counts()
    hit_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Hit ratio")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "hit_ratio.png", dpi=200)
    plt.close()

    # Hit rank histogram
    plt.figure()
    df_hits = df[df["hit_rank"].notna()]
    if not df_hits.empty:
        df_hits["hit_rank"].value_counts().sort_index().plot(kind="bar")
        plt.title("Hit rank distribution")
        plt.xlabel("Rank of first relevant source")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "hit_rank_hist.png", dpi=200)
    plt.close()

    # Top1 score histogram
    plt.figure()
    df["top1_score"].dropna().plot(kind="hist", bins=50)
    plt.title("Top-1 similarity score distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_score_hist.png", dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    report_dir = Path("report")
    plots_dir = report_dir / "plots"
    report_dir.mkdir(exist_ok=True)

    # Load eval set
    eval_set_path = Path("eval_set.jsonl")
    eval_set = load_eval_set(eval_set_path)

    top_ks = [1,3,5,10,15]
    embedding_models = ["all-MiniLM-L6-v2", "intfloat/e5-small", "LaBSE"]
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    all_results = []
    for model_name in embedding_models:
        CONFIG.embed_model_name = model_name
        out = evaluate(
            eval_set,
            embed_model_name=model_name,
            top_ks=top_ks,
            reranker=reranker,
            min_score=0.2
        )

        df = pd.DataFrame(out["rows"])
        df.to_csv(report_dir / f"eval_results_{model_name.replace('/', '_')}.csv", index=False, encoding="utf-8")

        # Compute summary metrics
        metrics = {"model": model_name, "n": out["n"]}
        metrics.update(out["recall_at_k"])
        metrics.update(out["mrr_at_k"])
        top1_scores = df["top1_score"].dropna()
        metrics.update({
            "top1_score_mean": float(top1_scores.mean()) if not top1_scores.empty else None,
            "top1_score_median": float(top1_scores.median()) if not top1_scores.empty else None,
            "top1_score_std": float(top1_scores.std()) if not top1_scores.empty else None,
        })
        for k in top_ks:
            metrics[f"hit_at_{k}"] = float((df["hit_rank"].fillna(999)<=k).mean())

        all_results.append(metrics)
        save_plots(df, plots_dir)

    # Save combined metrics
    df_cmp = pd.DataFrame(all_results)
    df_cmp.to_csv(report_dir / "metrics_comparison.csv", index=False, encoding="utf-8")
    print("\n✅ Experiment comparison table:")
    print(df_cmp[["model","recall_at_1","recall_at_3","recall_at_5","mrr_at_1","mrr_at_3","top1_score_mean"]])
    print(f"\nPlots saved in: {plots_dir.resolve()}")

if __name__ == "__main__":
    main()