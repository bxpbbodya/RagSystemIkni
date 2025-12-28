# pipelines/sync_all.py
from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional

from core.config import CONFIG
from pipelines.ingest_lpnu import ingest_lpnu_pages, DEFAULT_URLS
from pipelines.ingest_telegram import ingest_telegram_channel
from core.index import build_faiss_index, load_chunks_from_jsonl


def sync_lpnu() -> Dict[str, Any]:
    """
    Sync LPNU/Wiki pages into local cache (dedup inside ingest).
    """
    return ingest_lpnu_pages(
        urls=DEFAULT_URLS,
        cache_path=CONFIG.local_cache_path,
        chunk_size=900,
        overlap=120,
        polite_delay_sec=0.8,
    )


async def sync_telegram(
    api_id: int,
    api_hash: str,
    channels: List[str],
    *,
    limit: int = 300,
    since_days: Optional[int] = 120,
    session_name: str = "data/tg_session",
) -> List[Dict[str, Any]]:
    """
    Sync one or multiple Telegram channels into local cache.
    """
    results = []
    for ch in channels:
        r = await ingest_telegram_channel(
            api_id=api_id,
            api_hash=api_hash,
            channel=ch,
            cache_path=CONFIG.local_cache_path,
            limit=limit,
            since_days=since_days,
            chunk_size=600,
            overlap=80,
            session_name=session_name,
        )
        results.append(r)
    return results


def rebuild_index() -> Dict[str, Any]:
    """
    Rebuild FAISS index from local cache.
    """
    chunks = load_chunks_from_jsonl(CONFIG.local_cache_path)
    if not chunks:
        return {"ok": False, "error": "No chunks in local cache. Run sync first."}

    build_faiss_index(
        chunks=chunks,
        embed_model_name=CONFIG.embed_model_name,
        index_path=CONFIG.faiss_index_path,
        meta_path=CONFIG.faiss_meta_path,
    )
    return {"ok": True, "chunks_indexed": len(chunks)}


def _safe_asyncio_run(coro):
    """
    Safe asyncio runner for environments where an event loop might already be running.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # If we're already in an event loop (e.g., some environments), use alternative approach.
        if "asyncio.run()" in str(e) or "running event loop" in str(e).lower():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        raise


def sync_all(
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    channels: Optional[List[str]] = None,
    *,
    tg_limit: int = 300,
    tg_since_days: Optional[int] = 120,
    tg_session_name: str = "data/tg_session",
) -> Dict[str, Any]:
    """
    Sync LPNU + optional Telegram, then rebuild FAISS index.

    - LPNU sync always runs (can be disabled later if you want)
    - Telegram sync runs only if credentials and channels provided
    - Index rebuild runs if local cache contains chunks
    """
    report: Dict[str, Any] = {}

    # 1) Sync LPNU
    report["lpnu"] = sync_lpnu()

    # 2) Sync Telegram (optional)
    if api_id and api_hash and channels:
        report["telegram"] = _safe_asyncio_run(
            sync_telegram(
                api_id=api_id,
                api_hash=api_hash,
                channels=channels,
                limit=tg_limit,
                since_days=tg_since_days,
                session_name=tg_session_name,
            )
        )
    else:
        report["telegram"] = {
            "skipped": True,
            "reason": "No api_id/api_hash/channels provided",
        }

    # 3) Rebuild index
    report["index"] = rebuild_index()

    # 4) Useful totals for UI/reporting
    total_added = 0
    total_errors = 0

    # LPNU
    if isinstance(report.get("lpnu"), dict):
        total_added += int(report["lpnu"].get("added_chunks", 0))
        total_errors += len(report["lpnu"].get("errors", []) or [])

    # Telegram
    tg = report.get("telegram")
    if isinstance(tg, list):
        for r in tg:
            total_added += int(r.get("added_chunks", 0))
            total_errors += len(r.get("errors", []) or [])

    report["summary"] = {
        "total_added_chunks": total_added,
        "total_errors": total_errors,
        "index_ok": bool(report.get("index", {}).get("ok", False)),
    }

    return report
