from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import CONFIG
from core.index import build_faiss_index, load_chunks_from_jsonl
from core.kb_validation import validate_local_knowledge_base
from pipelines.ingest_telegram import ingest_telegram_channel
from pipelines.ingest_vns import ingest_vns_exports
from pipelines.ingest_site_resources import RESOURCE_URLS, ingest_resources

LPNU_CHUNK_SIZE = 180
LPNU_OVERLAP = 50
TELEGRAM_CHUNK_SIZE = 500
TELEGRAM_OVERLAP = 80
VNS_CHUNK_SIZE = 420
VNS_OVERLAP = 90
RESOURCE_CHUNK_SIZE = 420
RESOURCE_OVERLAP = 80


# -----------------------------
# LPNU Sync
# -----------------------------
async def _sync_lpnu() -> Dict[str, Any]:
    print("[LPNU] Starting crawl + ingest...")
    try:
        from pipelines.ingest_lpnu import _ensure_lpnu_ingest_deps, ingest_lpnu_async

        _ensure_lpnu_ingest_deps()
        result = await ingest_lpnu_async(
            seed_urls=None,
            cache_path=CONFIG.local_cache_path,
            chunk_size=LPNU_CHUNK_SIZE,
            overlap=LPNU_OVERLAP,
        )
        added_chunks = result.get("added_chunks", 0)
        total_pages = result.get("total_pages_crawled", 0)
        print(f"[LPNU] Added {added_chunks} chunks from {total_pages} pages")
        return {
            "ok": added_chunks > 0,
            "added_chunks": added_chunks,
            "processed_urls": total_pages,
            "pages_added": result.get("pages_added", []),
            "pages_skipped": result.get("pages_skipped", []),
            "chunks": [],
            "errors": [],
        }
    except Exception as e:
        return {
            "ok": False,
            "added_chunks": 0,
            "processed_urls": 0,
            "pages_added": [],
            "pages_skipped": [],
            "chunks": [],
            "errors": [str(e)],
        }


# -----------------------------
# VNS Sync (local exports)
# -----------------------------
def _sync_vns(export_dir: str | Path = "data/vns_exports") -> Dict[str, Any]:
    export_dir = Path(export_dir)
    if not export_dir.exists():
        return {
            "ok": False,
            "skipped": True,
            "reason": f"Directory not found: {export_dir}",
            "added_chunks": 0,
            "processed_files": 0,
            "errors": [],
        }

    result = ingest_vns_exports(
        export_dir=export_dir,
        cache_path=CONFIG.local_cache_path,
        chunk_size=VNS_CHUNK_SIZE,
        overlap=VNS_OVERLAP,
    )
    return result


def _sync_site_resources() -> Dict[str, Any]:
    return ingest_resources(
        RESOURCE_URLS,
        CONFIG.local_cache_path,
        chunk_size=RESOURCE_CHUNK_SIZE,
        overlap=RESOURCE_OVERLAP,
    )


# -----------------------------
# Telegram Sync
# -----------------------------
async def _sync_telegram(
    api_id: int,
    api_hash: str,
    channels: List[str],
    limit: int = 300,
    since_days: Optional[int] = 120,
    session_name: str = "data/tg_session",
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
                chunk_size=TELEGRAM_CHUNK_SIZE,
                overlap=TELEGRAM_OVERLAP,
                session_name=session_name,
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
def rebuild_index(chunks: Optional[List[Any]] = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = load_chunks_from_jsonl(CONFIG.local_cache_path)
    if not chunks:
        return {"ok": False, "error": "No chunks in cache. Run sync first."}

    quality_chunks = [ch for ch in chunks if len((ch.text or "").split()) >= 20]
    if not quality_chunks:
        return {"ok": False, "error": "No quality chunks available after filtering."}

    print(f"[INDEX] Rebuilding FAISS index with {len(quality_chunks)} chunks...")
    index, _ = build_faiss_index(
        chunks=quality_chunks,
        embed_model_name=CONFIG.embed_model_name,
        index_path=CONFIG.faiss_index_path,
        meta_path=CONFIG.faiss_meta_path,
    )
    return {"ok": True, "chunks_indexed": len(quality_chunks), "index_size": int(index.ntotal)}


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
    tg_session_name: str = "data/tg_session",
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    async def run_all():
        tasks = [_sync_lpnu()]
        if api_id and api_hash and channels:
            tasks.append(_sync_telegram(api_id, api_hash, channels, tg_limit, tg_since_days, tg_session_name))
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = _safe_asyncio_run(run_all())

    lpnu_result = results[0] if len(results) > 0 else {
        "ok": False,
        "added_chunks": 0,
        "processed_urls": 0,
        "pages_added": [],
        "pages_skipped": [],
        "chunks": [],
        "errors": ["Unknown error"],
    }
    report["lpnu"] = lpnu_result

    tg_result = results[1] if len(results) > 1 else []
    if isinstance(tg_result, Exception):
        report["telegram"] = {"ok": False, "added_chunks": 0, "errors": [str(tg_result)], "chunks": []}
    elif tg_result == []:
        report["telegram"] = {"skipped": True, "reason": "No api_id/api_hash/channels provided"}
    else:
        report["telegram"] = tg_result

    report["vns"] = _sync_vns()
    report["resources"] = _sync_site_resources()

    all_chunks = load_chunks_from_jsonl(CONFIG.local_cache_path)
    try:
        report["index"] = rebuild_index(all_chunks)
    except Exception as e:
        report["index"] = {"ok": False, "error": str(e)}
    report["kb_validation"] = validate_local_knowledge_base(CONFIG.local_cache_path)

    total_errors = len(lpnu_result.get("errors", []))
    if isinstance(tg_result, list):
        for r in tg_result:
            total_errors += len(r.get("errors", []))
    total_errors += len(report["vns"].get("errors", []))
    total_errors += len(report["resources"].get("errors", []))

    report["summary"] = {
        "total_chunks_in_cache": len(all_chunks),
        "lpnu_added_chunks": lpnu_result.get("added_chunks", 0),
        "lpnu_processed_urls": lpnu_result.get("processed_urls", 0),
        "vns_added_chunks": report["vns"].get("added_chunks", 0),
        "resource_added_chunks": report["resources"].get("added_chunks", 0),
        "eval_target_covered": report["kb_validation"].get("eval_target_covered", 0),
        "eval_target_total": report["kb_validation"].get("eval_target_total", 0),
        "total_errors": total_errors,
        "index_ok": report.get("index", {}).get("ok", False),
    }

    print(f"[SYNC SUMMARY] {report['summary']}")
    return report


if __name__ == "__main__":
    print("Starting full sync...")
    stats = sync_all()
    print(stats)
