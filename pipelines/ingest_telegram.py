# pipelines/ingest_telegram.py
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set

from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.types import Message

from core.ingest_utils import (
    make_chunks_from_doc,
    existing_chunk_ids as get_existing_chunk_ids,
    append_jsonl,
    load_jsonl,
)
from core.sources import SourceChunk


# -----------------------------
# Helpers
# -----------------------------
def _normalize_channel(channel: str) -> str:
    """
    Convert:
      - "https://t.me/pbikni" -> "pbikni"
      - "t.me/pbikni" -> "pbikni"
      - "@pbikni" -> "pbikni"
    """
    channel = (channel or "").strip()
    channel = channel.replace("https://t.me/", "").replace("http://t.me/", "")
    channel = channel.replace("t.me/", "")
    channel = channel.strip("@").strip("/").strip()
    return channel


def _iso_date(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.date().isoformat()


def _message_key(channel: str, msg_id: int) -> str:
    return f"tg:{channel}:{msg_id}"


def _safe_text(msg: Message) -> str:
    """
    Returns:
      - msg.message if present
      - caption of media if any
      - placeholder if media without text
    """
    txt = (getattr(msg, "message", None) or "").strip()
    if txt:
        return txt
    if getattr(msg, "media", None):
        return "[Media пост без тексту]"
    return ""


def _load_existing_message_keys(cache_path) -> Set[str]:
    """
    Scan local_cache.jsonl and collect message_key from extra.
    This makes dedup stable even if chunking changes slightly.
    """
    keys: Set[str] = set()
    try:
        items = load_jsonl(cache_path)
        for it in items:
            extra = it.get("extra") or {}
            mk = extra.get("message_key")
            if mk:
                keys.add(str(mk))
    except Exception:
        pass
    return keys


async def _with_timeout(coro, timeout_sec: int):
    """
    Prevent infinite hangs.
    """
    return await asyncio.wait_for(coro, timeout=timeout_sec)


# -----------------------------
# Main ingest
# -----------------------------
async def ingest_telegram_channel(
    *,
    api_id: int,
    api_hash: str,
    channel: str,
    cache_path,
    limit: int = 200,
    since_days: Optional[int] = 120,
    chunk_size: int = 600,
    overlap: int = 80,
    session_name: str = "data/tg_session",
    polite_delay_sec: float = 0.15,
    max_flood_wait_sec: int = 60,
    entity_timeout_sec: int = 20,
    iter_timeout_sec: int = 60,
) -> Dict[str, Any]:
    """
    Download last N messages from a Telegram channel and store in local cache JSONL.

    ✅ Important for MVP stability:
    - returns explicit error if session is NOT authorized
    - timeouts for entity + iteration to avoid infinite spinner
    - dedup by chunk_id + message_key
    """
    channel_norm = _normalize_channel(channel)

    # ✅ Dedup sets
    chunk_ids_seen = get_existing_chunk_ids(cache_path)
    message_keys_seen = _load_existing_message_keys(cache_path)

    added_chunks = 0
    processed_messages = 0
    skipped_messages = 0
    empty_messages = 0
    duplicate_messages = 0
    errors: List[str] = []

    since_dt: Optional[datetime] = None
    if since_days is not None and since_days > 0:
        since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)

    # Main client
    async with TelegramClient(session_name, api_id, api_hash) as client:
        # ✅ Ensure session is authorized (otherwise Telethon может ждать логін)
        try:
            is_auth = await _with_timeout(client.is_user_authorized(), timeout_sec=10)
            if not is_auth:
                return {
                    "source": "telegram",
                    "channel": channel_norm,
                    "processed_messages": 0,
                    "added_chunks": 0,
                    "skipped_messages": 0,
                    "empty_messages": 0,
                    "duplicate_messages": 0,
                    "errors": [
                        "Session is NOT authorized. "
                        "Run Telegram authorization first (phone + code) to create data/tg_session.session."
                    ],
                }
        except Exception as e:
            return {
                "source": "telegram",
                "channel": channel_norm,
                "processed_messages": 0,
                "added_chunks": 0,
                "skipped_messages": 0,
                "empty_messages": 0,
                "duplicate_messages": 0,
                "errors": [f"Authorization check failed: {e}"],
            }

        # ✅ Get channel entity with timeout
        try:
            entity = await _with_timeout(client.get_entity(channel_norm), timeout_sec=entity_timeout_sec)
        except Exception as e:
            return {
                "source": "telegram",
                "channel": channel_norm,
                "processed_messages": 0,
                "added_chunks": 0,
                "skipped_messages": 0,
                "empty_messages": 0,
                "duplicate_messages": 0,
                "errors": [f"get_entity failed: {e}"],
            }

        async def _iterate_messages():
            nonlocal added_chunks, processed_messages, skipped_messages, empty_messages, duplicate_messages

            async for msg in client.iter_messages(entity, limit=limit):
                processed_messages += 1

                if not isinstance(msg, Message):
                    skipped_messages += 1
                    continue

                # Date filter
                if since_dt and msg.date:
                    if msg.date < since_dt:
                        skipped_messages += 1
                        continue

                mk = _message_key(channel_norm, msg.id)
                if mk in message_keys_seen:
                    duplicate_messages += 1
                    continue

                raw_text = _safe_text(msg)
                if not raw_text:
                    empty_messages += 1
                    continue

                url = f"https://t.me/{channel_norm}/{msg.id}"
                title = f"Telegram: {channel_norm} / {msg.id}"
                date = _iso_date(msg.date)

                extra = {
                    "channel": channel_norm,
                    "msg_id": msg.id,
                    "message_key": mk,
                    "views": getattr(msg, "views", None),
                    "forwards": getattr(msg, "forwards", None),
                    "replies": getattr(getattr(msg, "replies", None), "replies", None),
                    "has_media": bool(getattr(msg, "media", None)),
                    "is_pinned": bool(getattr(msg, "pinned", False)),
                    "is_forward": bool(getattr(msg, "fwd_from", None)),
                }

                chunks: List[SourceChunk] = make_chunks_from_doc(
                    source_type="tg",
                    url=url,
                    title=title,
                    raw_text=raw_text,
                    date=date,
                    extra=extra,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                to_add = []
                for ch in chunks:
                    if ch.chunk_id in chunk_ids_seen:
                        continue
                    chunk_ids_seen.add(ch.chunk_id)
                    to_add.append(ch.__dict__)

                if to_add:
                    append_jsonl(cache_path, to_add)
                    added_chunks += len(to_add)

                # Add message key even if chunks were not added (future-proof)
                message_keys_seen.add(mk)

                if polite_delay_sec > 0:
                    await asyncio.sleep(polite_delay_sec)

        # ✅ Run iteration with timeout
        try:
            await _with_timeout(_iterate_messages(), timeout_sec=iter_timeout_sec)
        except FloodWaitError as e:
            wait_s = int(getattr(e, "seconds", 0) or 0)
            wait_s = min(wait_s, max_flood_wait_sec)
            errors.append(f"FloodWait: Telegram просить зачекати {wait_s}s. Зупиняю ingest (спробуй пізніше).")
        except asyncio.TimeoutError:
            errors.append(
                f"Timeout: iteration took more than {iter_timeout_sec}s. "
                "Reduce limit or increase iter_timeout_sec."
            )
        except RPCError as e:
            errors.append(f"Telegram RPCError: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

    return {
        "source": "telegram",
        "channel": channel_norm,
        "processed_messages": processed_messages,
        "added_chunks": added_chunks,
        "skipped_messages": skipped_messages,
        "empty_messages": empty_messages,
        "duplicate_messages": duplicate_messages,
        "errors": errors,
        "settings": {
            "limit": limit,
            "since_days": since_days,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "session_name": session_name,
            "polite_delay_sec": polite_delay_sec,
            "entity_timeout_sec": entity_timeout_sec,
            "iter_timeout_sec": iter_timeout_sec,
        },
    }
