# pipelines/ingest_lpnu.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import time
import re

import requests
import trafilatura
from bs4 import BeautifulSoup

from core.ingest_utils import (
    make_chunks_from_doc,
    existing_chunk_ids,
    append_jsonl,
    now_iso_date,
)


DEFAULT_URLS = [
    "https://lpnu.ua/ikni",
    "https://wiki.lpnu.ua/wiki/%D0%86%D0%BD%D1%81%D1%82%D0%B8%D1%82%D1%83%D1%82_%D0%BA%D0%BE%D0%BC%D0%BF%E2%80%99%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D0%B8%D1%85_%D0%BD%D0%B0%D1%83%D0%BA_%D1%82%D0%B0_%D1%96%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D1%96%D0%B9%D0%BD%D0%B8%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D1%96%D0%B9",
    "https://lpnu.ua/ikni/kerivnytstvo-instytutu",
    "https://lpnu.ua/ikni/oznaiomcha-informatsiia-pro-instytut",
    "https://lpnu.ua/ikni/vstupnyku-ikni",
    "https://lpnu.ua/ikni/vchena-rada-instytutu",
    "https://lpnu.ua/ikni/istoriia-instytutu",
    "https://lpnu.ua/ikni/partnery-instytutu",
    "https://lpnu.ua/ikni/kolehiia-ta-profbiuro-studentiv-instytutu",
    "https://lpnu.ua/ikni/prohramy-vstupnykh-vyprobuvan-mahistr",
    "https://lpnu.ua/ikni/naukova-robota-instytutu",
    "https://lpnu.ua/ikni/naukova-robota-instytutu/naukovi-ta-osvitni-proiekty",
    "https://lpnu.ua/ikni/naukova-robota-instytutu/naukovi-zhurnaly-ta-konferentsii",
    "https://lpnu.ua/ikni/informatsiia-dlia-studentiv-ta-abituriientiv",
    "https://lpnu.ua/ikni/polozhennia-pro-poriadok-provedennia-konkursu-stypendii-vid-blahodiinoho-fondu-klarina-ta",
]


# -----------------------------
# HTTP
# -----------------------------
def fetch_url(url: str, timeout: int = 25, retries: int = 2, sleep_sec: float = 0.7) -> str:
    headers = {
        "User-Agent": "IKNI-RAG-MVP/1.0 (edu project; contact: student)",
        "Accept-Language": "uk,en;q=0.8",
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text or ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_sec)

    raise RuntimeError(f"fetch_url failed after {retries+1} attempts: {last_err}")


# -----------------------------
# Extraction
# -----------------------------
def extract_text_trafilatura(html: str, url: str) -> str:
    try:
        downloaded = trafilatura.extract(
            html,
            url=url,
            include_links=False,
            include_images=False,
            favor_recall=False,
        )
        return (downloaded or "").strip()
    except Exception:
        return ""


def extract_text_bs4(html: str) -> str:
    """
    Fallback extractor (simple but stable).
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    return (text or "").strip()


def guess_title(html: str, fallback: str = "") -> str:
    m = re.search(r"<title>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return fallback
    title = (m.group(1) or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title or fallback


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) >= 2]

    cleaned = []
    for ln in lines:
        if ln.count("|") > 10 or ln.count("+") > 10:
            continue
        if re.search(r"\|\s*\|\s*\+", ln):
            continue
        cleaned.append(ln)

    text2 = "\n".join(cleaned)
    text2 = text2.replace("\xa0", " ")
    text2 = re.sub(r"[ \t]+", " ", text2)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()


# -----------------------------
# Relevance filter (cheap)
# -----------------------------
def is_relevant_page(text: str) -> bool:
    t = (text or "").lower()

    # short => skip
    if len(t) < 500:
        return False

    bad_markers = [
        "предметний тест з історії мистецтва",
        "мовознавства",
        "фольклорних фактів",
        "жанрів і стилів",
    ]
    if any(b in t for b in bad_markers):
        return False

    good_markers = ["ікні", "львівська політехніка", "інститут", "lpnu", "кафедр"]
    if not any(g in t for g in good_markers):
        return False

    return True


# -----------------------------
# Main pipeline
# -----------------------------
def ingest_lpnu_pages(
    urls: List[str],
    cache_path,
    *,
    chunk_size: int = 900,
    overlap: int = 120,
    polite_delay_sec: float = 0.8,
    fetch_timeout: int = 25,
) -> Dict[str, Any]:
    """
    Ingest LPNU pages into local_cache.jsonl.

    ✅ stable doc_id per URL: doc_id = f"lpnu::{url}"
    ✅ safe extraction + filtering
    ✅ safe dedup by chunk_id
    """
    cache_path = str(cache_path)
    existing_ids = existing_chunk_ids(cache_path)

    added_chunks = 0
    processed_urls = 0
    skipped_urls = 0
    errors: List[str] = []
    debug: List[Dict[str, Any]] = []

    for url in urls:
        processed_urls += 1
        url = (url or "").strip()
        if not url:
            skipped_urls += 1
            continue

        try:
            html = fetch_url(url, timeout=fetch_timeout, retries=2)
            title = guess_title(html, fallback=url)

            # extract
            text = extract_text_trafilatura(html, url=url)
            if not text or len(text) < 200:
                text = extract_text_bs4(html)

            # clean
            text = clean_extracted_text(text)

            # relevance
            if not is_relevant_page(text):
                skipped_urls += 1
                debug.append({"url": url, "skipped": True, "reason": "not_relevant_or_too_short"})
                time.sleep(polite_delay_sec)
                continue

            # ✅ stable doc_id per URL
            doc_id = f"lpnu::{url}"

            extra = {
                "origin": "lpnu_pages",
                "doc_id": doc_id,
                "seed_url": url,
            }

            chunks = make_chunks_from_doc(
                source_type="lpnu",
                url=url,
                title=title,
                raw_text=text,
                date=now_iso_date(),
                extra=extra,
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id=doc_id,  # ✅ force stable doc_id
            )

            to_add = []
            for ch in chunks:
                if not ch.chunk_id:
                    continue
                if ch.chunk_id in existing_ids:
                    continue
                existing_ids.add(ch.chunk_id)
                to_add.append(ch.__dict__)

            if to_add:
                append_jsonl(cache_path, to_add)
                added_chunks += len(to_add)

            debug.append({"url": url, "chunks": len(chunks), "added": len(to_add)})

        except Exception as e:
            errors.append(f"{url} -> {e}")

        time.sleep(polite_delay_sec)

    return {
        "source": "lpnu",
        "processed_urls": processed_urls,
        "added_chunks": added_chunks,
        "skipped_urls": skipped_urls,
        "errors": errors,
        "debug": debug[-30:],  # keep last 30
    }
