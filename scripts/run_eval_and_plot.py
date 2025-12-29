from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt

from core.config import CONFIG
from core.index import load_faiss_index, search_index


# -----------------------------
# Simple retrieval eval
# -----------------------------
def is_hit(chunk, rule: dict) -> bool:
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


def evaluate(eval_set: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)

    hits = 0
    rr_sum = 0.0
    rows = []

    for ex in eval_set:
        query = ex["query"]
        results = search_index(
            query=query,
            index=index,
            chunks=meta,
            embed_model_name=CONFIG.embed_model_name,
            top_k=top_k,
        )

        hit_rank: Optional[int] = None
        for i, (chunk, score) in enumerate(results, start=1):
            if is_hit(chunk, ex):
                hit_rank = i
                break

        hit = hit_rank is not None
        if hit:
            hits += 1
            rr_sum += 1.0 / hit_rank

        rows.append({
            "query": query,
            "hit": hit,
            "hit_rank": hit_rank,
            "top1_score": float(results[0][1]) if results else None,
            "top1_url": results[0][0].url if results else None,
        })

    n = len(eval_set)
    recall = hits / n if n else 0
    mrr = rr_sum / n if n else 0

    return {
        "n": n,
        "top_k": top_k,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "rows": rows,
    }


# -----------------------------
# Plotting helpers
# -----------------------------
def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Hit rate pie
    hit_counts = df["hit"].value_counts()
    plt.figure()
    hit_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Evaluation: Hit ratio")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "hit_ratio.png", dpi=200)
    plt.close()

    # 2) Rank histogram (only hits)
    plt.figure()
    df_hits = df[df["hit"] == True]
    if not df_hits.empty:
        df_hits["hit_rank"].value_counts().sort_index().plot(kind="bar")
        plt.title("Hit rank distribution")
        plt.xlabel("Rank of first relevant source")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "hit_rank_hist.png", dpi=200)
    plt.close()

    # 3) Top1 score histogram
    plt.figure()
    df["top1_score"].dropna().plot(kind="hist", bins=10)
    plt.title("Top-1 similarity score distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_score_hist.png", dpi=200)
    plt.close()



def main():
    eval_path = Path("eval_set.jsonl")
    if not eval_path.exists():
        raise FileNotFoundError("eval_set.jsonl not found")

    eval_set = []
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            eval_set.append(json.loads(line))
        except Exception:
            continue

    report_dir = Path("report")
    plots_dir = report_dir / "plots"
    report_dir.mkdir(exist_ok=True)

    out = evaluate(eval_set, top_k=5)
    df = pd.DataFrame(out["rows"])

    # ✅ EXTRA METRICS
    hit1 = float((df["hit_rank"] == 1).mean())
    hit3 = float((df["hit_rank"].fillna(999) <= 3).mean())
    hit5 = float((df["hit_rank"].fillna(999) <= 5).mean())

    mean_top1 = float(df["top1_score"].dropna().mean()) if df["top1_score"].notna().any() else None
    median_top1 = float(df["top1_score"].dropna().median()) if df["top1_score"].notna().any() else None

    # save metrics
    metrics = {
        "n": out["n"],
        "top_k": out["top_k"],
        "recall_at_k": out["recall_at_k"],
        "mrr_at_k": out["mrr_at_k"],
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "top1_score_mean": mean_top1,
        "top1_score_median": median_top1,
    }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    df.to_csv(report_dir / "eval_results.csv", index=False, encoding="utf-8")

    save_plots(df, plots_dir)

    print("✅ DONE")
    print(metrics)
    print(f"Plots saved in: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
