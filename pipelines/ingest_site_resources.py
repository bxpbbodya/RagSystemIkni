# pipelines/ingest_site_resources.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import re
import requests
from bs4 import BeautifulSoup
import trafilatura

from core.ingest_utils import (
    make_chunks_from_doc,
    existing_chunk_ids,
    append_jsonl,
    now_iso_date,
)

# -----------------------------
# DEFAULT RESOURCE URLS
# -----------------------------
RESOURCE_URLS = [
    "https://lpnu.ua/news",
    "https://lpnu.ua/novyny-fakultetiv",
    "https://lpnu.ua/osvitni-prohramy",
    "https://lpnu.ua/normatyvni-dokumenty",
    "https://lpnu.ua/aktyvnist-studentiv",
    # можна додати інші факультети/інститути
]

# -----------------------------
# Fetch page with retries
# -----------------------------
def fetch_url(url: str, timeout: int = 20, retries: int = 2, delay: float = 0.5) -> str:
    headers = {
        "User-Agent": "LPNU-RAG-RESOURCES/1.0",
        "Accept-Language": "uk,en;q=0.8",
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay)
    print(f"[WARN] Failed to fetch {url}: {last_err}")
    return ""

# -----------------------------
# Text extraction
# -----------------------------
def extract_text(html: str, url: str) -> str:
    text = trafilatura.extract(html, url=url, include_links=False, include_images=False)
    if text and len(text) > 100:
        return text.strip()
    # fallback
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main = soup.find("main")
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return text.strip() if text else ""

def guess_title(html: str, fallback: str = "") -> str:
    m = re.search(r"<title>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return fallback

def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 2]
    cleaned = [ln for ln in lines if ln.count("|") <= 10 and ln.count("+") <= 10]
    text2 = "\n".join(cleaned)
    text2 = text2.replace("\xa0", " ")
    text2 = re.sub(r"[ \t]+", " ", text2)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()

def is_relevant(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 200:
        return False
    return True

# -----------------------------
# Main ingestion function
# -----------------------------
def ingest_resources(
    urls: List[str],
    cache_path,
    *,
    chunk_size: int = 650,
    overlap: int = 120,
    polite_delay: float = 0.4,
    fetch_timeout: int = 20,
) -> Dict[str, Any]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = existing_chunk_ids(cache_path)

    added_chunks = 0
    processed_urls = 0
    skipped_urls = 0
    errors: List[str] = []
    debug: List[Dict[str, Any]] = []

    for url in urls:
        processed_urls += 1
        url = url.strip()
        if not url:
            skipped_urls += 1
            continue

        html = fetch_url(url, timeout=fetch_timeout)
        if not html:
            skipped_urls += 1
            continue

        title = guess_title(html, fallback=url)
        text = extract_text(html, url=url)
        text = clean_text(text)

        if not is_relevant(text):
            skipped_urls += 1
            debug.append({"url": url, "skipped": True, "reason": "too_short_or_bad_content"})
            time.sleep(polite_delay)
            continue

        doc_id = f"lpnu_resource::{url}"
        extra = {
            "origin": "lpnu_resource",
            "doc_id": doc_id,
            "seed_url": url,
            "source_trust": 0.86,
            "content_type": "html",
            "word_count_est": len(text.split()),
            "version": now_iso_date(),
        }

        chunks = make_chunks_from_doc(
            source_type="lpnu_resource",
            url=url,
            title=title,
            raw_text=text,
            date=now_iso_date(),
            extra=extra,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
        )

        to_add = []
        for ch in chunks:
            if not ch.chunk_id or ch.chunk_id in existing_ids:
                continue
            existing_ids.add(ch.chunk_id)
            to_add.append(ch.__dict__)

        if to_add:
            append_jsonl(cache_path, to_add)
            added_chunks += len(to_add)

        debug.append({"url": url, "chunks": len(chunks), "added": len(to_add)})
        time.sleep(polite_delay)

    return {
        "source": "lpnu_resources",
        "processed_urls": processed_urls,
        "added_chunks": added_chunks,
        "skipped_urls": skipped_urls,
        "errors": errors,
        "debug": debug[-30:],
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cache_file = "data/resources_cache.jsonl"
    stats = ingest_resources(RESOURCE_URLS, cache_file)
    print("[INFO] Resource ingestion stats:", stats)
