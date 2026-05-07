from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.ingest_utils import (
    append_jsonl,
    existing_chunk_ids,
    existing_doc_ids,
    make_chunks_from_doc,
    now_iso_date,
)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import docx
except Exception:
    docx = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".pdf", ".docx"}


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_html_file(path: Path) -> str:
    raw = _read_text_file(path)
    if BeautifulSoup is None:
        return raw
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup
    return main.get_text("\n", strip=True)


def _collect_json_text(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _collect_json_text(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _collect_json_text(value)
    elif isinstance(obj, (str, int, float)):
        text = str(obj).strip()
        if len(text) >= 3:
            yield text


def _read_json_file(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        texts: List[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    texts.append(line)
                    continue
                texts.extend(_collect_json_text(obj))
        return "\n".join(texts)

    obj = json.loads(_read_text_file(path))
    return "\n".join(_collect_json_text(obj))


def _read_pdf_file(path: Path) -> str:
    if fitz is None:
        return ""
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _read_docx_file(path: Path) -> str:
    if docx is None:
        return ""
    document = docx.Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def read_vns_export(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _read_text_file(path)
    if suffix in {".html", ".htm"}:
        return _read_html_file(path)
    if suffix in {".json", ".jsonl"}:
        return _read_json_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    if suffix == ".docx":
        return _read_docx_file(path)
    return ""


def ingest_vns_exports(
    export_dir: str | Path = "data/vns_exports",
    cache_path: str | Path = "data/local_cache.jsonl",
    *,
    chunk_size: int = 420,
    overlap: int = 90,
) -> Dict[str, Any]:
    export_dir = Path(export_dir)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not export_dir.exists():
        return {"ok": False, "added_chunks": 0, "processed_files": 0, "errors": [f"Directory not found: {export_dir}"]}

    existing_ids = existing_chunk_ids(cache_path)
    existing_docs = existing_doc_ids(cache_path)

    added_chunks = 0
    processed_files = 0
    added_files: List[str] = []
    skipped_files: List[str] = []
    errors: List[str] = []

    for path in sorted(export_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        processed_files += 1
        rel = path.relative_to(export_dir).as_posix()
        doc_id = f"vns::{rel}"
        if doc_id in existing_docs:
            skipped_files.append(rel)
            continue

        try:
            raw_text = _normalize_text(read_vns_export(path))
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
            continue

        if len(raw_text.split()) < 20:
            skipped_files.append(rel)
            continue

        title = path.stem.replace("_", " ").replace("-", " ").strip()
        stat = path.stat()
        extra = {
            "origin": "vns_export",
            "doc_id": doc_id,
            "relative_path": rel,
            "content_type": path.suffix.lower().lstrip("."),
            "source_trust": 0.95,
            "version": int(stat.st_mtime),
            "file_modified_at": stat.st_mtime,
            "word_count_est": len(raw_text.split()),
        }
        chunks = make_chunks_from_doc(
            source_type="vns",
            url=f"vns://{rel}",
            title=title or rel,
            raw_text=raw_text,
            date=now_iso_date(),
            extra=extra,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
        )

        to_add = []
        for chunk in chunks:
            if not chunk.chunk_id or chunk.chunk_id in existing_ids:
                continue
            existing_ids.add(chunk.chunk_id)
            to_add.append(chunk.__dict__)

        if not to_add:
            skipped_files.append(rel)
            continue

        append_jsonl(cache_path, to_add)
        added_chunks += len(to_add)
        added_files.append(rel)
        existing_docs.add(doc_id)

    return {
        "ok": added_chunks > 0,
        "added_chunks": added_chunks,
        "processed_files": processed_files,
        "added_files": added_files,
        "skipped_files": skipped_files,
        "errors": errors,
    }
