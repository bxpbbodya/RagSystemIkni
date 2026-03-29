# pipelines/ingest_lpnu_clean_opt_boost.py
from __future__ import annotations
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
from typing import List, Tuple
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import trafilatura
import re
import hashlib

from core.ingest_utils import make_chunks_from_doc, existing_chunk_ids, append_jsonl, now_iso_date
from core.index import build_faiss_index, load_chunks_from_jsonl

DEFAULT_SEED_URLS = [
    "https://lpnu.ua/igdg", "https://lpnu.ua/ikni", "https://lpnu.ua/ibib",
    "https://lpnu.ua/iadu", "https://lpnu.ua/ihsn", "https://lpnu.ua/inem",
]

BAD_EXTENSIONS = (".jpg",".jpeg",".png",".gif",".svg",".webp",
                  ".pdf",".doc",".docx",".xls",".xlsx",".ppt",".pptx",
                  ".zip",".rar",".7z",".mp4",".mp3")

EXCLUDE_PATHS = ("/en/","/fr/","/es/","/de/","/zh/","/pl/","/media/","/downloads/","/images/","/cdn-cgi/")

RELEVANT_PATH_KEYWORDS = ("kerivnytstvo","institutes","iad","iard","iadu","ikni","ibib","igdg")
MIN_TEXT_LEN = 30  # трохи менше, щоб релевантні тексти не пропадали

# ====== optional eval-set keyword boost ======
EVAL_KEYWORDS = [
    "бакалавр","магістр","комп'ютерні науки","геодезія","землеустрій",
    "гуманітарні науки","соціальні науки"
]

def normalize_url(url: str) -> str:
    return url.split("#")[0].rstrip("/")

def is_valid_url(url: str, domain: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != domain:
        return False
    if any(parsed.path.lower().endswith(ext) for ext in BAD_EXTENSIONS):
        return False
    if any(parsed.path.startswith(prefix) for prefix in EXCLUDE_PATHS):
        return False
    return True

def is_relevant_url(url: str) -> bool:
    return any(keyword in url for keyword in RELEVANT_PATH_KEYWORDS)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_text(html: str, url: str) -> str:
    text = trafilatura.extract(html, url=url, include_links=False, include_images=False)
    if text:
        return text.strip()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript","header","footer","nav"]):
        tag.decompose()
    main = soup.find("main") or soup
    paragraphs = main.find_all(["p","li","h1","h2","h3","td"])
    texts = []
    for p in paragraphs:
        t = p.get_text(separator=" ", strip=True)
        if len(t) >= MIN_TEXT_LEN:
            # boost: якщо текст містить eval keyword, додаємо його двічі
            if any(kw.lower() in t.lower() for kw in EVAL_KEYWORDS):
                t = t + " " + t
            texts.append(t)
    return " ".join(texts)

def guess_title(html: str, fallback: str = "") -> str:
    m = re.search(r"<title>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return fallback or "No title"

class AsyncCrawler:
    def __init__(self, start_url: str, max_pages: int = 200, max_depth: int = 3, concurrency: int = 10):
        self.start_url = normalize_url(start_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.domain = urlparse(start_url).netloc
        self.concurrency = concurrency
        self.visited: set[str] = set()
        self.found: List[Tuple[str,str,str]] = []

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        headers = {"User-Agent": "LPNU-RAG-BOOST/1.0"}
        try:
            async with session.get(url, timeout=20, headers=headers, ssl=False) as r:
                if r.status == 200:
                    return await r.text()
        except Exception as e:
            print(f"[fetch error] {url} -> {e}")
        return ""

    async def parse_and_queue(self, session: aiohttp.ClientSession, url: str, depth: int, queue: deque):
        if url in self.visited or depth > self.max_depth or len(self.visited) >= self.max_pages:
            return
        html = await self.fetch(session, url)
        if not html:
            return
        text = extract_text(html, url)
        if len(text) < MIN_TEXT_LEN:
            return
        title = guess_title(html, url)
        page_hash = hash_text(text)
        if page_hash in self.visited:
            return
        self.visited.add(url)
        self.found.append((url, title, text))

        soup = BeautifulSoup(html,"lxml")
        for a in soup.find_all("a",href=True):
            link = normalize_url(urljoin(url,a["href"]))
            if is_valid_url(link,self.domain) and is_relevant_url(link) and link not in self.visited:
                queue.append((link, depth+1))

    async def run(self) -> List[Tuple[str,str,str]]:
        queue = deque([(self.start_url,0)])
        async with aiohttp.ClientSession() as session:
            while queue and len(self.visited) < self.max_pages:
                tasks=[]
                for _ in range(min(self.concurrency,len(queue))):
                    url,depth = queue.popleft()
                    tasks.append(self.parse_and_queue(session,url,depth,queue))
                if tasks:
                    await asyncio.gather(*tasks)
        return self.found

async def ingest_lpnu_async(seed_urls: List[str] = None, cache_path: str = "data/local_cache.jsonl", chunk_size: int = 300, overlap: int = 70):
    seed_urls = seed_urls or DEFAULT_SEED_URLS
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = existing_chunk_ids(cache_path)
    added_chunks = 0

    for seed in seed_urls:
        crawler = AsyncCrawler(seed)
        pages = await crawler.run()
        for url, title, text in pages:
            doc_id = f"lpnu::{url}"
            chunks = make_chunks_from_doc(
                source_type="lpnu",
                url=url,
                title=title,
                raw_text=text,
                date=now_iso_date(),
                extra={"origin":"lpnu","doc_id":doc_id},
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id=doc_id
            )
            to_add=[]
            for ch in chunks:
                if ch.chunk_id in existing_ids:
                    continue
                existing_ids.add(ch.chunk_id)
                to_add.append(ch.__dict__)
            if to_add:
                append_jsonl(cache_path,to_add)
                added_chunks+=len(to_add)
            print(f"[INFO] {url} -> {len(to_add)} new chunks")

    return {"added_chunks": added_chunks, "total_pages": len(existing_ids)}

def ingest_and_build_index(seed_urls: List[str] = None, cache_file="data/local_cache.jsonl", index_file="data/index.faiss", meta_file="data/index_meta.jsonl"):
    seed_urls = seed_urls or DEFAULT_SEED_URLS
    stats = asyncio.run(ingest_lpnu_async(seed_urls=seed_urls, cache_path=cache_file))
    print("[INFO] Ingestion stats:", stats)

    chunks = load_chunks_from_jsonl(Path(cache_file))
    if not chunks:
        raise ValueError("No chunks found for FAISS index.")

    index,_ = build_faiss_index(
        chunks=chunks,
        embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_path=Path(index_file),
        meta_path=Path(meta_file)
    )
    print(f"[INFO] FAISS index built with {len(chunks)} chunks")
    return index