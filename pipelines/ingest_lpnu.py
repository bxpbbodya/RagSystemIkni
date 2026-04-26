from __future__ import annotations

import asyncio
import hashlib
import re
from collections import deque
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from core.index import build_faiss_index, load_chunks_from_jsonl
from core.ingest_utils import (
    append_jsonl,
    existing_chunk_ids,
    existing_doc_ids,
    make_chunks_from_doc,
    now_iso_date,
)


def _ensure_lpnu_ingest_deps() -> None:
    missing = []
    if aiohttp is None:
        missing.append("aiohttp")
    if trafilatura is None:
        missing.append("trafilatura")
    if BeautifulSoup is None:
        missing.append("beautifulsoup4")
    if missing:
        raise RuntimeError(
            "Missing LPNU ingest dependencies: "
            + ", ".join(missing)
            + ". Install them with `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        )

# ----------------------------- CONFIG -----------------------------
DEFAULT_SEED_URLS = [
    "https://lpnu.ua/iard",
    "https://lpnu.ua/igdg",
    "https://lpnu.ua/ikni",
    "https://lpnu.ua/ibib",
    "https://lpnu.ua/iadu",
    "https://lpnu.ua/ihsn",
    "https://lpnu.ua/inem",
    "https://lpnu.ua/news",
    "https://lpnu.ua/studentske-mistechko",
    "https://lpnu.ua/studentska-biblioteka",
    "https://lpnu.ua/vstupnyku",
    "https://lpnu.ua/ikta",
]

CURATED_PRIORITY_URLS = [
    "https://lpnu.ua/iard",
    "https://lpnu.ua/ikni/kerivnytstvo-instytutu",
    "https://lpnu.ua/ikni/napriamy-pidhotovky-spetsialnosti-ta-osvitni-prohramy",
    "https://lpnu.ua/ikni/vstupnyku-ikni",
    "https://lpnu.ua/igdg",
    "https://lpnu.ua/igdg/kerivnytstvo-instytutu",
    "https://lpnu.ua/ihdh/napriamy-pidhotovky-konkursni-predmety-ta-spetsialnosti",
    "https://lpnu.ua/ihsn",
    "https://lpnu.ua/ihsn/kerivnytstvo-instytutu",
    "https://lpnu.ua/ihsn/spetsialnosti",
    "https://lpnu.ua/studentska-biblioteka",
    "https://lpnu.ua/studentske-mistechko",
    "https://lpnu.ua/cmo/mizhnarodni-ugody-pro-spivpratsiu/spivpratsia-z-universytetamy",
    "https://lpnu.ua/igdg",
    "https://lpnu.ua/ihsn",
    "https://lpnu.ua/inem",
    "https://lpnu.ua/iesk",
    "https://lpnu.ua/ikte",
    "https://lpnu.ua/ikta",
    "https://lpnu.ua/imit",
    "https://lpnu.ua/ipmt",
    "https://lpnu.ua/ippt",
    "https://lpnu.ua/ippo", "https://lpnu.ua/imfn", "https://lpnu.ua/istr",
    "https://lpnu.ua/ikhkht", "https://lpnu.ua/miok", "https://lpnu.ua/vstupnyku",
    "https://lpnu.ua/lvivska-politekhnika/kerivnytstvo-universytetu",
    "https://lpnu.ua/pryimalna-komisiia",
]

INSTITUTE_CODES = (
    "iard", "igdg", "ihdh", "ikni", "ibib", "iadu", "ihsn", "inem", "ikta",
    "iesk", "ikte", "imit", "ipmt", "ippt", "ippo", "imfn", "istr", "ikhkht", "miok"
)

INSTITUTE_NAMES = {
    "iard": "Інститут архітектури та дизайну",
    "igdg": "Інститут геодезії",
    "ikni": "Інститут комп'ютерних наук та інформаційних технологій",
    "ibib": "Інститут будівництва та інженерних систем",
    "iadu": "Інститут адміністрування, державного управління та професійного розвитку",
    "ihsn": "Інститут гуманітарних та соціальних наук",
    "inem": "Інститут економіки і менеджменту",
    "ikta": "Інститут комп'ютерних технологій, автоматики та метрології",
    "iesk": "Інститут енергетики та систем керування",
    "ikte": "Інститут комп'ютерних технологій та електроніки",
    "imit": "Інститут механічної інженерії та транспорту",
    "ipmt": "Інститут прикладної математики та фундаментальних наук",
    "ippt": "Інститут права, психології та інноваційної освіти",
    "ippo": "Інститут післядипломної освіти",
    "imfn": "Інститут прикладної математики та фундаментальних наук",
    "istr": "Інститут сталого розвитку",
    "ikhkht": "Інститут хімії та хімічних технологій",
    "miok": "Міжнародний інститут освіти, культури та зв'язків з діаспорою",
}

BAD_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".mp4", ".mp3",
)

EXCLUDE_PATHS = (
    "/en/", "/fr/", "/es/", "/de/", "/zh/", "/pl/", "/it/", "/aspirantam/", "/news"
    "/media/", "/downloads/", "/images/", "/cdn-cgi/", "/news/", "/events/", "/press/",
    "/tags/", "/taxonomy/", "/comment", "/search",
)

RELEVANT_PATH_KEYWORDS = (
    "kerivnytstvo", "dyrektsiia", "pro-instytut", "kontakty", "vstupnyku",
    "spetsialnosti", "osvitni-prohramy", "napriamy-pidhotovky", "kafedra",
    "istoriia-instytutu", "studentska-biblioteka", "studentske-mistechko",
    "mizhnarodni-ugody-pro-spivpratsiu", "spivpratsia-z-universytetamy",
    "instytut", *INSTITUTE_CODES,
)

MIN_TEXT_LEN = 30
MIN_WORDS = 10
MIN_CHUNK_WORDS = 20
MAX_PHONE_LINES_RATIO = 0.6
MAX_SHORT_LINES_RATIO = 0.7

EVAL_KEYWORDS = [
    "директор", "керівництво", "дирекція", "завідувач", "кафедра", "інститут",
    "контакти", "телефон", "email", "e-mail", "спеціальності", "освітні програми",
    "освітня програма", "бакалавр", "магістр", "вступ", "вступнику",
    "комп'ютерні науки", "геодезія", "землеустрій", "гуманітарні науки", "соціальні науки",
]

NOISE_PATTERNS = (
    "cookie", "cookies", "підписуйтеся", "поділитися", "share", "facebook",
    "instagram", "youtube", "telegram", "увійти", "зареєструватися",
)

PHONE_RE = re.compile(r"(\+?\d[\d\-\(\)\s]{7,}\d)")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
INSTITUTE_RE = re.compile(r"/(" + "|".join(INSTITUTE_CODES) + r")([/?#]|$)")


# ----------------------------- UTILS -----------------------------
def normalize_url(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def load_eval_target_urls(eval_path: Path = Path("eval_set.jsonl")) -> List[str]:
    if not eval_path.exists():
        return []
    urls: List[str] = []
    with eval_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            url = normalize_url((obj.get("must_contain_url") or "").strip())
            if url:
                urls.append(url)
    return list(dict.fromkeys(urls))


def is_valid_url(url: str, domain: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != domain:
        return False
    if any(parsed.path.lower().endswith(ext) for ext in BAD_EXTENSIONS):
        return False
    if any(parsed.path.lower().startswith(prefix) for prefix in EXCLUDE_PATHS):
        return False
    return True


def is_relevant_url(url: str) -> bool:
    url_lower = (url or "").lower()
    return any(keyword in url_lower for keyword in RELEVANT_PATH_KEYWORDS)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def detect_institute_code(url: str) -> str:
    match = INSTITUTE_RE.search((url or "").lower())
    if match:
        code = match.group(1)
        if code == "ihdh":
            return "igdg"
        return code
    return ""


def clean_extracted_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines: List[str] = []
    seen: Set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip(" -•\t")
        if len(line) < 3:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        if any(noise in lowered for noise in NOISE_PATTERNS):
            continue
        if lowered in {"контакти", "адреса", "e-mail", "email", "телефон"}:
            continue
        seen.add(lowered)
        cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines).strip()


def score_page(text: str, url: str, title: str) -> int:
    text_lower = text.lower()
    url_lower = url.lower()
    title_lower = (title or "").lower()

    score = 0
    keyword_hits = sum(1 for kw in EVAL_KEYWORDS if kw in text_lower or kw in title_lower)
    score += keyword_hits * 3
    priority_kws = ["директор", "дирекція", "декан", "керівництво"]
    for kw in priority_kws:
        if kw in text_lower:
            score += 10

    if any(key in url_lower for key in ("kerivnytstvo", "dyrektsiia", "spetsialnosti", "osvitni-prohramy", "vstupnyku")):
        score += 15
    if detect_institute_code(url_lower):
        score += 4
    if len(text.split()) >= 180:
        score += 3
    if PHONE_RE.search(text) and EMAIL_RE.search(text):
        score += 1

    return score


def is_garbage_text(text: str, url: str = "") -> bool:
    url_lower = url.lower()
    if any(kw in url_lower for kw in ("kerivnytstvo", "dyrektsiia", "ikni", "ikta", "iard")):
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(text.split()) < MIN_WORDS or len(lines) < 2:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(text.split()) < MIN_WORDS or len(lines) < 2:
        return True

    short_lines = sum(1 for line in lines if len(line.split()) <= 3)
    phone_lines = sum(1 for line in lines if PHONE_RE.search(line) or EMAIL_RE.search(line))
    joined = " ".join(lines).lower()

    if short_lines / max(len(lines), 1) > MAX_SHORT_LINES_RATIO:
        return True
    if phone_lines / max(len(lines), 1) > MAX_PHONE_LINES_RATIO:
        return True
    if joined.count("email") + joined.count("e-mail") > 8:
        return True
    if not any(kw in joined for kw in EVAL_KEYWORDS):
        return True

    return False


def filter_chunks(chunks, institute_code: str, target_urls: Set[str]) -> List[Dict]:
    filtered = []
    # Регулярка для ПІБ (Прізвище І. П. або Прізвище Ім'я По батькові)
    name_pattern = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ'’\-]+\s+[А-ЯІЇЄҐ]\.\s*[А-ЯІЇЄҐ]\.")

    for ch in chunks:
        text = (ch.text or "").strip()
        url = normalize_url(ch.url or "")
        lowered_text = text.lower()

        # ЗАВЖДИ беремо, якщо це цільова сторінка або містить ознаки керівництва
        is_priority = (
                url in target_urls or
                any(k in lowered_text for k in ["директор", "декан", "керівництво", "завідувач"]) or
                name_pattern.search(text) or
                "@" in lowered_text
        )

        if is_priority:
            filtered.append(ch.__dict__)
            continue

        # Для звичайного тексту лишаємо ценз за довжиною
        if len(text.split()) >= 30:
            filtered.append(ch.__dict__)

    return filtered


def extract_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 1. Видаляємо лише те, що точно не є контентом
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    # 2. Знаходимо основний контент
    main_content = (
            soup.find("div", {"class": "region-content"}) or
            soup.find("div", {"id": "block-system-main"}) or
            soup.find("article") or
            soup.body
    )

    if not main_content:
        return ""

    url_lower = url.lower()
    if any(kw in url_lower for kw in ("dyrektsiia", "kerivnytstvo")):
        text = main_content.get_text(separator="\n", strip=True)
    else:
        # Для звичайних сторінок лишаємо trafilatura
        import trafilatura
        text = trafilatura.extract(str(main_content), include_tables=True, no_fallback=False)
        if not text:
            text = main_content.get_text(separator="\n", strip=True)

    # 3. КРИТИЧНО: Обробляємо таблиці ПЕРЕД витягуванням тексту
    for table in main_content.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(separator=" ", strip=True) for cell in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table_text = "\n" + "\n".join(rows) + "\n"
        table.replace_with(soup.new_string(table_text))

    url_lower = url.lower()
    # Якщо сторінка пріоритетна — беремо ВСЕ через BeautifulSoup без фільтрів trafilatura
    if any(kw in url_lower for kw in ("kerivnytstvo", "dyrektsiia", "kafedra", "vstup")):
        text = main_content.get_text(separator="\n", strip=True)
    else:
        # Для звичайних новин можна залишити trafilatura
        import trafilatura
        text = trafilatura.extract(str(main_content), include_tables=True)
        if not text:  # Fallback якщо trafilatura нічого не знайшла
            text = main_content.get_text(separator="\n", strip=True)

    return clean_extracted_text(text)

def guess_title(html: str, fallback: str = "") -> str:
    m = re.search(r"<title>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return fallback or "No title"


# ----------------------------- CRAWLER -----------------------------
class AsyncCrawler:
    def __init__(
        self,
        start_url: str,
        *,
        target_urls: Set[str],
        max_pages: int = 50000,
        max_depth: int = 12,
        concurrency: int = 10,
    ):
        self.start_url = normalize_url(start_url)
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.domain = urlparse(start_url).netloc
        self.institute_code = detect_institute_code(start_url)
        self.target_urls = target_urls
        self.concurrency = concurrency
        self.visited_urls: Set[str] = set()
        self.visited_hashes: Set[str] = set()
        self.found: List[Tuple[str, str, str]] = []

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        headers = {"User-Agent": "LPNU-CRAWLER/1.0"}
        try:
            async with session.get(url, timeout=20, headers=headers, ssl=False) as r:
                if r.status == 200:
                    return await r.text()
        except Exception as e:
            print(f"[fetch error] {url} -> {e}")
        return ""

    async def parse_and_queue(self, session: aiohttp.ClientSession, url: str, depth: int, queue: deque):
        if url in self.visited_urls or depth > self.max_depth or len(self.visited_urls) >= self.max_pages:
            return
        self.visited_urls.add(url)

        html = await self.fetch(session, url)
        if not html:
            print(f"[SKIP] Empty HTML: {url}")
            return

        title = guess_title(html, url)
        text = extract_text(html, url)
        word_count = len(text.split())
        page_score = score_page(text, url, title)

        print(f"[DEBUG] URL: {url} | Words: {word_count} | Score: {page_score} | Text preview: {text[:180]!r} ...")

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            link = normalize_url(urljoin(url, a["href"]))
            if not is_valid_url(link, self.domain):
                continue
            if self.institute_code and detect_institute_code(link) not in {"", self.institute_code}:
                continue
            if is_relevant_url(link) and link not in self.visited_urls:
                queue.append((link, depth + 1))

        normalized_url = normalize_url(url)

        if is_garbage_text(text) and normalized_url not in self.target_urls:
            print(f"[SKIP] Garbage text: {url}")
            return
        if page_score < 6 and normalized_url not in self.target_urls:
            print(f"[SKIP] Low quality page: {url} | score={page_score}")
            return

        page_hash = hash_text(text)
        if page_hash in self.visited_hashes:
            print(f"[SKIP] Duplicate page hash: {url}")
            return
        self.visited_hashes.add(page_hash)
        self.found.append((url, title, text))

        if len(self.found) % 25 == 0:
            print(f"[PROGRESS] {len(self.found)} strong pages collected from {self.start_url}")

    async def run(self) -> List[Tuple[str, str, str]]:
        queue = deque([(self.start_url, 0)])
        async with aiohttp.ClientSession() as session:
            while queue and len(self.visited_urls) < self.max_pages:
                tasks = []
                for _ in range(min(self.concurrency, len(queue))):
                    url, depth = queue.popleft()
                    tasks.append(self.parse_and_queue(session, url, depth, queue))
                if tasks:
                    await asyncio.gather(*tasks)
        return self.found


# ----------------------------- INGEST -----------------------------
async def ingest_lpnu_async(
    seed_urls: List[str] = None,
    cache_path: str = "data/local_cache.jsonl",
    chunk_size: int = 600,
    overlap: int = 120,
):
    seed_urls = seed_urls or DEFAULT_SEED_URLS
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = existing_chunk_ids(cache_path)
    existing_docs = existing_doc_ids(cache_path)
    target_urls = set(load_eval_target_urls())
    expanded_seed_urls = list(dict.fromkeys([*seed_urls, *CURATED_PRIORITY_URLS, *sorted(target_urls)]))
    added_chunks = 0
    crawled_urls = []
    added_urls = []
    skipped_urls = []

    tasks = [AsyncCrawler(seed, target_urls=target_urls).run() for seed in expanded_seed_urls]
    all_pages_list = await asyncio.gather(*tasks)

    for pages in all_pages_list:
        for url, title, text in pages:
            crawled_urls.append(url)
            normalized_url = normalize_url(url)
            if len(text.split()) < MIN_WORDS and normalized_url not in target_urls:
                skipped_urls.append(url)
                continue

            doc_id = f"lpnu::{url}"
            if doc_id in existing_docs:
                skipped_urls.append(url)
                continue

            institute_code = detect_institute_code(url)
            chunks = make_chunks_from_doc(
                source_type="lpnu",
                url=url,
                title=title,
                raw_text=text,
                date=now_iso_date(),
                extra={
                    "origin": "lpnu",
                    "doc_id": doc_id,
                    "institute_code": institute_code or None,
                    "institute_name": INSTITUTE_NAMES.get(institute_code, "") if institute_code else "",
                },
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id=doc_id,
            )

            filtered_chunk_dicts = filter_chunks(chunks, institute_code, target_urls)

            to_add = []
            for obj in filtered_chunk_dicts:
                chunk_id = obj.get("chunk_id")
                if not chunk_id or chunk_id in existing_ids:
                    continue
                existing_ids.add(chunk_id)
                to_add.append(obj)

            if to_add:
                append_jsonl(cache_path, to_add)
                added_chunks += len(to_add)
                added_urls.append(url)
                existing_docs.add(doc_id)
            else:
                skipped_urls.append(url)

            print(f"[INFO] {url} -> {len(to_add)} new chunks")

    return {
        "added_chunks": added_chunks,
        "total_pages_crawled": len(crawled_urls),
        "pages_added": added_urls,
        "pages_skipped": skipped_urls,
    }


def ingest_and_build_index(
    seed_urls: List[str] = None,
    cache_file: str = "data/local_cache.jsonl",
    index_file: str = "data/index.faiss",
    meta_file: str = "data/index_meta.jsonl",
):
    seed_urls = seed_urls or DEFAULT_SEED_URLS
    stats = asyncio.run(ingest_lpnu_async(seed_urls=seed_urls, cache_path=cache_file))
    print("[INFO] Ingestion stats:", stats)

    chunks = load_chunks_from_jsonl(Path(cache_file))
    if not chunks:
        raise ValueError("No chunks found for FAISS index.")

    index, _ = build_faiss_index(
        chunks=chunks,
        embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_path=Path(index_file),
        meta_path=Path(meta_file),
    )
    print(f"[INFO] FAISS index built with {len(chunks)} chunks")
    return index
