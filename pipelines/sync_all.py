# pipelines/sync_all_optimized.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.config import CONFIG
from core.index import build_faiss_index, load_chunks_from_jsonl
from core.ingest_utils import make_chunks_from_doc, append_jsonl, existing_chunk_ids, now_iso_date

from pipelines.ingest_lpnu import AsyncCrawler, ingest_lpnu_async
from pipelines.ingest_telegram import ingest_telegram_channel


# -----------------------------
# LPNU Sync via AsyncCrawler
# -----------------------------
async def _sync_lpnu_async() -> Dict[str, Any]:
    print("Starting LPNU crawl + ingest...")
    crawler = AsyncCrawler(
        start_url="https://lpnu.ua",
        max_pages=100,
        max_depth=2,
        concurrency=5
    )
    pages = await crawler.run()

    if not pages:
        return {"ok": False, "added_chunks": 0, "processed_urls": 0, "chunks": [], "errors": ["No pages crawled"]}

    existing_ids = existing_chunk_ids(CONFIG.local_cache_path)
    added_chunks = 0
    all_chunks = []

    for url, title, text in pages:
        doc_id = f"lpnu::{url}"
        chunks = make_chunks_from_doc(
            source_type="lpnu",
            url=url,
            title=title,
            raw_text=text,
            date=now_iso_date(),
            extra={"origin": "lpnu", "doc_id": doc_id},
            chunk_size=900,
            overlap=120,
            doc_id=doc_id
        )

        to_add = []
        for ch in chunks:
            if ch.chunk_id in existing_ids:
                continue
            existing_ids.add(ch.chunk_id)
            to_add.append(ch.__dict__)
            all_chunks.append(ch.__dict__)

        if to_add:
            append_jsonl(CONFIG.local_cache_path, to_add)
            added_chunks += len(to_add)

        print(f"[LPNU CHUNKS] {url} -> {len(to_add)} added")

    return {"ok": added_chunks > 0, "added_chunks": added_chunks, "processed_urls": len(pages), "chunks": all_chunks, "errors": []}


# -----------------------------
# Telegram Sync
# -----------------------------
async def _sync_telegram(
    api_id: int,
    api_hash: str,
    channels: List[str],
    limit: int = 300,
    since_days: Optional[int] = 120,
    session_name: str = "data/tg_session"
) -> List[Dict[str, Any]]:
    results = []
    for ch in channels:
        try:
            r = await ingest_telegram_channel(
                api_id=api_id,
                api_hash=api_hash,
                channel=ch,
                cache_path=CONFIG.local_cache_path,
                limit=limit,
                since_days=since_days,
                chunk_size=600,
                overlap=80,
                session_name=session_name
            )
            added = len(r.get("chunks", []))
            errors = r.get("errors", [])
            print(f"[TG] {ch}: {added} chunks, {len(errors)} errors")
            results.append({"channel": ch, "added_chunks": added, "errors": errors, "chunks": r.get("chunks", [])})
        except Exception as e:
            print(f"[TG ERROR] {ch}: {e}")
            results.append({"channel": ch, "added_chunks": 0, "errors": [str(e)], "chunks": []})
    return results


# -----------------------------
# FAISS Index Rebuild
# -----------------------------
def rebuild_index(chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = load_chunks_from_jsonl(CONFIG.local_cache_path)
    if not chunks:
        return {"ok": False, "error": "No chunks in cache. Run sync first."}

    print(f"Rebuilding FAISS index with {len(chunks)} chunks...")
    index, _ = build_faiss_index(
        chunks=chunks,
        embed_model_name=CONFIG.embed_model_name,
        index_path=CONFIG.faiss_index_path,
        meta_path=CONFIG.faiss_meta_path
    )
    return {"ok": True, "chunks_indexed": len(chunks), "index_size": int(index.ntotal)}


# -----------------------------
# Safe asyncio runner
# -----------------------------
def _safe_asyncio_run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "running event loop" in str(e).lower():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


# -----------------------------
# Full Sync Pipeline
# -----------------------------
def sync_all(
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    channels: Optional[List[str]] = None,
    tg_limit: int = 300,
    tg_since_days: Optional[int] = 120,
    tg_session_name: str = "data/tg_session"
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # LPNU
    try:
        lpnu_result = _safe_asyncio_run(_sync_lpnu_async())
        report["lpnu"] = lpnu_result
    except Exception as e:
        report["lpnu"] = {"ok": False, "added_chunks": 0, "processed_urls": 0, "chunks": [], "errors": [str(e)]}

    # Telegram
    tg_result: List[Dict[str, Any]] = []
    if api_id and api_hash and channels:
        tg_result = _safe_asyncio_run(_sync_telegram(api_id, api_hash, channels, tg_limit, tg_since_days, tg_session_name))
        report["telegram"] = tg_result
    else:
        report["telegram"] = {"skipped": True, "reason": "No api_id/api_hash/channels provided"}

    # Merge all chunks for FAISS
    all_chunks = []
    if lpnu_result.get("chunks"):
        all_chunks.extend(lpnu_result["chunks"])
    if isinstance(tg_result, list):
        for r in tg_result:
            if r.get("chunks"):
                all_chunks.extend(r["chunks"])

    # Rebuild FAISS index
    try:
        report["index"] = rebuild_index(all_chunks)
    except Exception as e:
        report["index"] = {"ok": False, "error": str(e)}

    # Summary
    total_added = len(all_chunks)
    total_errors = len(lpnu_result.get("errors", []))
    if isinstance(tg_result, list):
        for r in tg_result:
            total_errors += len(r.get("errors", []))

    report["summary"] = {
        "total_added_chunks": total_added,
        "total_errors": total_errors,
        "index_ok": report.get("index", {}).get("ok", False)
    }

    print(f"[SYNC SUMMARY] {report['summary']}")
    return report


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    print("Starting full sync...")
    stats = sync_all()
    print(stats)