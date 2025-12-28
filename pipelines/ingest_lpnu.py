# pipelines/ingest_lpnu.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import time
import re

import requests
import trafilatura
from bs4 import BeautifulSoup

from core.ingest_utils import make_chunks_from_doc, existing_chunk_ids, append_jsonl, now_iso_date
from core.sources import SourceChunk


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


def fetch_url(url: str, timeout: int = 25) -> Optional[str]:
    headers = {
        "User-Agent": "IKNI-RAG-MVP/1.0 (edu project; contact: student)",
        "Accept-Language": "uk,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def extract_text_trafilatura(html: str, url: str) -> str:
    downloaded = trafilatura.extract(
        html,
        url=url,
        include_links=False,
        include_images=False,
        favor_recall=False,
    )
    return downloaded or ""


def extract_text_bs4(html: str) -> str:
    """
    Fallback extractor (simple, but stable).
    """
    soup = BeautifulSoup(html, "lxml")

    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # try main content first
    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    return text or ""


def guess_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    title = m.group(1).strip()
    title = re.sub(r"\s+", " ", title)
    return title


def clean_extracted_text(text: str) -> str:
    """
    Remove common junk patterns from extracted text.
    """
    if not text:
        return ""

    # Remove repeated menu-ish lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) >= 2]

    # Drop very noisy lines
    cleaned = []
    for ln in lines:
        if ln.count("|") > 10 or ln.count("+") > 10:
            continue
        if re.search(r"\|\s*\|\s*\+", ln):
            continue
        cleaned.append(ln)

    text2 = "\n".join(cleaned)

    # Normalize whitespace
    text2 = text2.replace("\xa0", " ")
    text2 = re.sub(r"[ \t]+", " ", text2)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()


def is_relevant_page(text: str) -> bool:
    """
    Very light relevance gate to avoid obvious unrelated garbage.
    """
    t = text.lower()

    # If page is too short after cleaning, skip it
    if len(t) < 500:
        return False

    # Filter out unrelated "test" / "art history" etc. (sometimes appears in wiki pages or embedded blocks)
    bad_markers = [
        "предметний тест з історії мистецтва",
        "мовознавства",
        "фольклорних фактів",
        "жанрів і стилів",  # obviously not about IKNI
    ]
    if any(b in t for b in bad_markers):
        return False

    # Must contain at least one IKNI/LPN indicators
    good_markers = ["ікні", "львівська політехніка", "інститут", "lpnu", "кафедр"]
    if not any(g in t for g in good_markers):
        return False

    return True


def ingest_lpnu_pages(
    urls: List[str],
    cache_path,
    chunk_size: int = 900,
    overlap: int = 120,
    polite_delay_sec: float = 0.8,
) -> Dict[str, Any]:
    existing_ids = existing_chunk_ids(cache_path)
    added = 0
    processed = 0
    skipped = 0
    errors: List[str] = []

    for url in urls:
        processed += 1
        try:
            html = fetch_url(url)
            title = guess_title(html) or url

            # Extract
            text = extract_text_trafilatura(html, url=url)
            if not text or len(text.strip()) < 200:
                text = extract_text_bs4(html)

            # Clean
            text = clean_extracted_text(text)

            # Relevance gate
            if not is_relevant_page(text):
                skipped += 1
                time.sleep(polite_delay_sec)
                continue

            chunks: List[SourceChunk] = make_chunks_from_doc(
                source_type="lpnu",
                url=url,
                title=title,
                raw_text=text,
                date=now_iso_date(),
                extra={"origin": "lpnu_pages"},
                chunk_size=chunk_size,
                overlap=overlap,
            )

            to_add = []
            for ch in chunks:
                if ch.chunk_id in existing_ids:
                    continue
                existing_ids.add(ch.chunk_id)
                to_add.append(ch.__dict__)

            if to_add:
                append_jsonl(cache_path, to_add)
                added += len(to_add)

        except Exception as e:
            errors.append(f"{url} -> {e}")

        time.sleep(polite_delay_sec)

    return {
        "source": "lpnu",
        "processed_urls": processed,
        "added_chunks": added,
        "skipped_urls": skipped,
        "errors": errors,
    }
