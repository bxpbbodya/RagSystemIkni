import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
import hashlib

EVAL_PAGES = [
    "https://lpnu.ua/ikni/kerivnytstvo-instytutu",
    "https://lpnu.ua/ikni/napriamy-pidhotovky-spetsialnosti-ta-osvitni-prohramy",
    "https://lpnu.ua/ikni/vstupnyku-ikni",
    "https://lpnu.ua/igdg/kerivnytstvo-instytutu",
    "https://lpnu.ua/igdg/napriamy-pidhotovky-konkursni-predmety-ta-spetsialnosti",
    "https://lpnu.ua/ihsn/kerivnytstvo-instytutu",
    "https://lpnu.ua/ihsn/spetsialnosti",
    "https://lpnu.ua/studentske-mistechko",
    "https://lpnu.ua/news",
    "https://lpnu.ua/cmo/mizhnarodni-ugody-pro-spivpratsiu/spivpratsia-z-universytetamy",
    "https://lpnu.ua/studentska-biblioteka",
]

MIN_TEXT_LEN = 30  # трохи менше для релевантних chunk

EVAL_KEYWORDS = [
    "бакалавр","магістр","комп'ютерні науки","геодезія","землеустрій",
    "гуманітарні науки","соціальні науки"
]

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

class EvalCrawler:
    def __init__(self, urls, concurrency=10):
        self.urls = urls
        self.semaphore = asyncio.Semaphore(concurrency)
        self.text_hashes = set()
        self.found = []

    async def fetch(self, session: ClientSession, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 LPNU-EVAL-BOOST/1.0"}
        try:
            async with self.semaphore:
                async with session.get(url, timeout=20, headers=headers, ssl=False) as r:
                    if r.status == 200:
                        html = await r.text()
                        print(f"[OK] {url} | len={len(html)}")
                        return html
        except Exception as e:
            print(f"[ERROR] {url} -> {e}")
        return ""

    async def parse_page(self, html: str, url: str):
        soup = BeautifulSoup(html, "lxml")
        content_div = soup.find("main") or soup.find("div", class_="content") or soup.find("article") or soup
        paragraphs = content_div.find_all(["p", "li", "h1", "h2", "h3", "td"])
        texts = []
        for p in paragraphs:
            txt = p.get_text(separator=" ", strip=True)
            if len(txt) >= MIN_TEXT_LEN:
                # boost: дублюємо абзаци з eval keywords
                if any(kw.lower() in txt.lower() for kw in EVAL_KEYWORDS):
                    txt = txt + " " + txt
                texts.append(txt)
        # сортуємо абзаци з eval keywords на початок
        texts.sort(key=lambda t: any(kw.lower() in t.lower() for kw in EVAL_KEYWORDS), reverse=True)
        full_text = " ".join(texts)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        if len(full_text) < MIN_TEXT_LEN:
            print(f"[SKIP SHORT] {url} | len={len(full_text)}")
            return None
        text_md5 = hash_text(full_text)
        if text_md5 in self.text_hashes:
            print(f"[SKIP DUPLICATE] {url}")
            return None
        self.text_hashes.add(text_md5)
        title = soup.title.string.strip() if soup.title else url
        return {"url": url, "title": title, "text": full_text}

    async def fetch_and_parse(self, session, url):
        html = await self.fetch(session, url)
        if html:
            return await self.parse_page(html, url)
        return None

    async def crawl(self):
        async with ClientSession() as session:
            tasks = [self.fetch_and_parse(session, url) for url in self.urls]
            results = await asyncio.gather(*tasks)
            self.found = [r for r in results if r]
        return self.found

async def main():
    crawler = EvalCrawler(EVAL_PAGES, concurrency=10)
    results = await crawler.crawl()
    print(f"\n=== Total pages crawled: {len(results)} ===")
    for page in results:
        print(f"{page['url']} | {page['title'][:80]} | text_len: {len(page['text'])}")
        filename = re.sub(r'\W+', '_', page['title'])[:50] + ".txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(page['text'])

if __name__ == "__main__":
    asyncio.run(main())