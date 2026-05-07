# app.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import json
import zipfile

import streamlit as st
import pandas as pd

# matplotlib safe backend (important for Windows / streamlit)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import CONFIG
from core.index import load_faiss_index, build_faiss_index, load_chunks_from_jsonl
from core.security import mask_secret
from core.llm import LLMSettings, chat_completion, build_base_url

from core.query_pipeline import AnswerQueryConfig, answer_query_with_config

# Optional reranker
try:
    from core.rerank import rerank_results
    RERANK_AVAILABLE = True
except Exception:
    rerank_results = None
    RERANK_AVAILABLE = False

FEEDBACK_PATH = Path("data/feedback.jsonl")


# ==========================================================
# Page config
# ==========================================================
st.set_page_config(page_title=CONFIG.project_name, layout="wide")


# ==========================================================
# Safe JSONL helpers (DO NOT CRASH on broken lines)
# ==========================================================
def safe_read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # skip broken line
                    continue
    except Exception:
        return []

    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def safe_append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ==========================================================
# Feedback
# ==========================================================
def _append_feedback(entry: dict) -> None:
    safe_append_jsonl(FEEDBACK_PATH, entry)


def _feedback_payload(rating: int, comment: str = "") -> dict:
    query = st.session_state.get("last_query", "")
    ans = st.session_state.get("last_answer_struct")
    results = st.session_state.get("last_results", [])

    answer_text = ""
    used_sources: List[int] = []
    warnings: List[str] = []

    if ans:
        answer_text = (getattr(ans, "markdown", "") or "")[:1200]
        used_sources = list(getattr(ans, "used_sources", []) or [])
        warnings = list(getattr(ans, "warnings", []) or [])

    sources = []
    for rank, (chunk, score) in enumerate(results[:10], start=1):
        extra = chunk.extra or {}
        sources.append({
            "rank": rank,
            "score": float(score),
            "title": chunk.title,
            "url": chunk.url,
            "source_type": chunk.source_type,
            "date": chunk.date,
            "chunk_id": chunk.chunk_id,
            "doc_id": extra.get("doc_id"),
        })

    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "rating": int(rating),
        "comment": (comment or "").strip(),
        "query": query,
        "answer_snippet": answer_text,
        "used_sources": used_sources,
        "warnings": warnings,
        "retrieval_sources": sources,
        "online_mode": bool(st.session_state.get("online_mode")),
        "llm_enabled": bool(st.session_state.get("llm_enabled")),
        "llm_provider": st.session_state.get("llm_provider"),
        "llm_model": st.session_state.get("llm_model"),
        "reranker_enabled": bool(st.session_state.get("use_reranker_ui")),
        "min_score": float(st.session_state.get("min_score", 0.0)),
        "keyword_filter": bool(st.session_state.get("keyword_filter", True)),
        "doc_scope_enabled": bool(st.session_state.get("doc_scope_enabled", False)),
        "doc_scope_ids": sorted(list(st.session_state.get("doc_scope_ids", set()) or [])),
    }
    return payload


# ==========================================================
# Model presets
# ==========================================================
MODEL_PRESETS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openrouter": [
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-2.0-flash-exp",
        "anthropic/claude-3.5-sonnet",
    ],
    "ollama": [
        "llama3.1",
        "mistral",
        "qwen2.5",
    ],
    "custom": [],
}
PROVIDERS = ["openai", "groq", "openrouter", "ollama", "custom"]


# ==========================================================
# Session state
# ==========================================================
def _init_state():
    defaults = {
        # Online mode
        "online_mode": True,

        # RAG state
        "index_ready": False,
        "last_results": [],
        "last_answer_struct": None,
        "last_query": "",
        "last_sync_report": None,

        # LLM
        "llm_enabled": False,
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "",
        "llm_base_url": "",
        "llm_temperature": 0.2,
        "llm_debug": False,
        "use_custom_model": False,

        # Retrieval tuning
        "min_score": 0.35,
        "keyword_filter": True,
        "show_retrieval_debug": False,
        "use_structured_extraction": True,
        "use_institute_filter": True,
        "use_semantic_rules": True,

        # Reranker
        "use_reranker_ui": bool(getattr(CONFIG, "use_reranker", False)) and RERANK_AVAILABLE,
        "reranker_model_ui": getattr(CONFIG, "reranker_model_name", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "reranker_top_n_ui": int(getattr(CONFIG, "reranker_top_n", 30)),

        # Source filters
        "filter_lpnu": True,
        "filter_tg": True,
        "filter_vns": True,
        "filter_local": True,

        # Document scope (local uploads)
        "doc_scope_enabled": False,
        "doc_scope_ids": set(),

        # Answer UI
        "show_used_sources_only": True,
        "show_chunk_preview": False,

        # quick query
        "quick_query": "",

        # Feedback
        "feedback_comment": "",

        # Telegram
        "tg_api_id": "",
        "tg_api_hash": "",
        "tg_channels": "pbikni",
        "tg_phone": "",
        "tg_code": "",
        "tg_2fa": "",

        # VNS (UI only)
        "vns_login": "",
        "vns_password": "",

        # Eval state
        "last_eval_metrics": None,
        "last_eval_df": None,
        "last_eval_plots_dir": None,
        "last_benchmark_summary": None,
        "last_benchmark_details": None,
        "last_kb_validation": None,
        "last_self_improve_report": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ==========================================================
# Helpers
# ==========================================================
def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _parse_tg_channels(raw: str) -> List[str]:
    chans: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            chans.append(line)
    return chans


def _normalize_channel(channel: str) -> str:
    ch = (channel or "").strip()
    ch = ch.replace("https://t.me/", "").replace("http://t.me/", "").replace("t.me/", "")
    ch = ch.strip("@").strip("/").strip()
    return ch


def _maybe_load_index() -> None:
    try:
        load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
        st.session_state.index_ready = True
    except Exception:
        st.session_state.index_ready = False


def _online_badge() -> None:
    if st.session_state.online_mode:
        st.sidebar.success("🟢 ONLINE — доступ до інтернету дозволено")
    else:
        st.sidebar.error("🔴 OFFLINE — тільки локальні дані")


def _build_llm_settings() -> LLMSettings:
    return LLMSettings(
        enabled=bool(st.session_state.llm_enabled and st.session_state.online_mode),
        provider=st.session_state.llm_provider,
        model=(st.session_state.llm_model or "").strip(),
        api_key=(st.session_state.llm_api_key or "").strip(),
        base_url=(st.session_state.llm_base_url or "").strip() or None,
        temperature=float(st.session_state.llm_temperature),
        max_tokens=650,
    )


def _current_allowed_types() -> Optional[Tuple[str, ...]]:
    allowed_types = []
    if st.session_state.filter_lpnu:
        allowed_types.append("lpnu")
    if st.session_state.filter_tg:
        allowed_types.append("tg")
    if st.session_state.filter_vns:
        allowed_types.append("vns")
    if st.session_state.filter_local:
        allowed_types.append("local")
    return tuple(allowed_types) if allowed_types else None


def _current_allowed_doc_ids() -> Optional[Tuple[str, ...]]:
    if st.session_state.doc_scope_enabled and st.session_state.doc_scope_ids:
        return tuple(sorted(st.session_state.doc_scope_ids))
    return None


def _build_current_query_config(
    *,
    top_k: int,
    llm: Optional[LLMSettings],
    use_institute_filter: Optional[bool] = None,
) -> AnswerQueryConfig:
    return AnswerQueryConfig(
        embed_model_name=CONFIG.embed_model_name,
        min_score=float(st.session_state.min_score),
        keyword_filter=bool(st.session_state.keyword_filter),
        allowed_types=_current_allowed_types(),
        allowed_doc_ids=_current_allowed_doc_ids(),
        use_reranker=bool(st.session_state.use_reranker_ui and RERANK_AVAILABLE),
        reranker_model=st.session_state.reranker_model_ui,
        reranker_top_n=int(st.session_state.reranker_top_n_ui),
        use_query_expansion=True,
        use_adaptive_top_k=True,
        use_hybrid=True,
        top_k_override=int(top_k),
        use_post_boosts=True,
        use_extraction=bool(st.session_state.use_structured_extraction),
        use_institute_filter=bool(st.session_state.use_institute_filter if use_institute_filter is None else use_institute_filter),
        use_rules=bool(st.session_state.use_semantic_rules),
        llm=llm,
    )


def _provider_default_model(provider: str) -> str:
    presets = MODEL_PRESETS.get(provider) or []
    return presets[0] if presets else ""


def _ensure_valid_model_for_provider(provider: str) -> None:
    presets = MODEL_PRESETS.get(provider) or []
    if provider == "custom":
        return
    if presets and st.session_state.llm_model not in presets:
        st.session_state.llm_model = presets[0]


def _delete_file_silent(p: Path) -> bool:
    try:
        if p.exists():
            p.unlink()
        return True
    except Exception:
        return False


def _wipe_local_storage() -> dict:
    ok = True
    ok &= _delete_file_silent(Path(CONFIG.local_cache_path))
    ok &= _delete_file_silent(Path(CONFIG.faiss_index_path))
    ok &= _delete_file_silent(Path(CONFIG.faiss_meta_path))
    return {"ok": ok}


def export_report_zip() -> Optional[Path]:
    report_dir = Path("report")
    if not report_dir.exists():
        return None

    zip_path = report_dir / "report_package.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in report_dir.rglob("*"):
                if p.is_dir():
                    continue
                if p.name.endswith(".zip"):
                    continue
                z.write(p, arcname=str(p.relative_to(report_dir)))
        return zip_path
    except Exception:
        return None


# -------- Telegram async runner ----------
def _run_async(coro, timeout_sec: int = 25):
    import asyncio
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout_sec))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout_sec))
        finally:
            loop.close()


# ==========================================================
# Evaluation helpers (FIXED metrics + no undefined vars)
# ==========================================================
def _load_eval_set(path: Path) -> List[dict]:
    items = safe_read_jsonl(path)
    return items


def _is_hit(chunk, rule: dict) -> bool:
    url = (chunk.url or "").lower()
    title = (chunk.title or "").lower()
    stype = (chunk.source_type or "").lower()
    text = (chunk.text or "").lower()

    must_url = (rule.get("must_contain_url") or "").lower()
    must_type = (rule.get("must_contain_type") or "").lower()
    must_text = (rule.get("must_contain_text") or "").lower()

    if must_url and must_url not in url:
        return False
    if must_type and must_type != stype:
        return False
    if must_text and must_text not in text and must_text not in title:
        return False
    return True


def _save_eval_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hit_counts = df["hit"].value_counts()
    plt.figure()
    hit_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Evaluation: Hit ratio")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dir / "hit_ratio.png", dpi=200)
    plt.close()

    plt.figure()
    df_hits = df[df["hit"] == True]
    if not df_hits.empty:
        df_hits["hit_rank"].value_counts().sort_index().plot(kind="bar")
        plt.title("Hit rank distribution")
        plt.xlabel("Rank of first relevant source")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "hit_rank_hist.png", dpi=200)
    plt.close()

    plt.figure()
    df["top1_score"].dropna().plot(kind="hist", bins=10)
    plt.title("Top-1 similarity score distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_score_hist.png", dpi=200)
    plt.close()

from pathlib import Path
import pandas as pd
import json
from typing import Optional

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

# -----------------------------
# ROUGE-L
# -----------------------------
def rouge_l_score(pred: str, ref: str) -> float:
    if not pred or not ref or rouge_scorer is None:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores['rougeL'].fmeasure

# -----------------------------
# Semantic similarity (cosine)
# -----------------------------
_embed_cache = {}
def semantic_similarity_score(pred: str, ref: str, embed_model: str = "all-MiniLM-L6-v2") -> float:
    if not pred or not ref:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return 0.0
    try:
        if embed_model not in _embed_cache:
            try:
                _embed_cache[embed_model] = SentenceTransformer(embed_model, local_files_only=True)
            except Exception:
                _embed_cache[embed_model] = SentenceTransformer(embed_model)
        model = _embed_cache[embed_model]
        embeddings = model.encode([pred, ref], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return float(sim)
    except Exception:
        return 0.0

# -----------------------------
# Run retrieval evaluation
# -----------------------------
def run_retrieval_eval(top_k: int = 5, use_reranker: bool = False, use_institute_filter: bool = True) -> dict:
    from core.evaluation import PipelineConfig, load_eval_set, run_pipeline_evaluation

    eval_path = Path("eval_set.jsonl")
    eval_set = load_eval_set(eval_path)
    if not eval_set:
        return {"ok": False, "error": "eval_set.jsonl not found or empty."}

    index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
    llm = _build_llm_settings() if (st.session_state.online_mode and st.session_state.llm_enabled) else None
    base_query_config = _build_current_query_config(
        top_k=top_k,
        llm=llm,
        use_institute_filter=use_institute_filter,
    )
    config = PipelineConfig(
        name="app_eval_pipeline",
        embed_model=base_query_config.embed_model_name,
        top_k=top_k,
        allowed_types=base_query_config.allowed_types,
        use_hybrid=base_query_config.use_hybrid,
        use_post_boosts=base_query_config.use_post_boosts,
        use_reranker=bool(use_reranker and RERANK_AVAILABLE),
        reranker_model=st.session_state.reranker_model_ui if (use_reranker and RERANK_AVAILABLE) else None,
        reranker_top_n=base_query_config.reranker_top_n,
        use_query_expansion=base_query_config.use_query_expansion,
        use_adaptive_top_k=base_query_config.use_adaptive_top_k,
        use_generation=bool(llm and llm.enabled),
        use_extraction=base_query_config.use_extraction,
        use_institute_filter=base_query_config.use_institute_filter,
        use_rules=base_query_config.use_rules,
        keyword_filter=base_query_config.keyword_filter,
        min_score=base_query_config.min_score,
    )
    metrics, df = run_pipeline_evaluation(
        eval_set=eval_set,
        index=index,
        chunks=meta,
        config=config,
        llm=llm,
    )
    metrics["mode"] = "adaptive_hybrid_reranker" if (use_reranker and RERANK_AVAILABLE) else "adaptive_hybrid"
    metrics["min_score"] = float(st.session_state.min_score)
    metrics["keyword_filter"] = bool(st.session_state.keyword_filter)
    metrics["use_institute_filter"] = bool(use_institute_filter)
    metrics["reranker_model"] = st.session_state.reranker_model_ui if (use_reranker and RERANK_AVAILABLE) else None
    metrics["reranker_top_n"] = int(st.session_state.reranker_top_n_ui) if (use_reranker and RERANK_AVAILABLE) else None

    report_dir = Path("report")
    plots_dir = report_dir / "plots"
    report_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(report_dir / "eval_results.csv", index=False, encoding="utf-8")
        (report_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        errors = list(getattr(df, "attrs", {}).get("errors", []) or [])
        (report_dir / "eval_errors.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in errors),
            encoding="utf-8",
        )
        _save_eval_plots(df, plots_dir)
    except Exception:
        pass

    return {
        "ok": True,
        "metrics": metrics,
        "df": df,
        "plots_dir": str(plots_dir)
    }

def dataset_stats_from_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"ok": False, "error": "local_cache.jsonl not found"}

    types: Dict[str, int] = {}
    dates: Dict[str, int] = {}

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            stype = obj.get("source_type", "other")
            if stype:
                types[stype] = types.get(stype, 0) + 1

            d = obj.get("date")
            if d:
                dates[d] = dates.get(d, 0) + 1

    return {"ok": True, "types": types, "dates": dates}


# ==========================================================
# Mode switch (VERY IMPORTANT)
# ==========================================================
mode = st.sidebar.radio(
    "Режим",
    ["LIGHT", "PRO"],
    index=0
)
if mode == "PRO":
    # ==========================================================
    # Sidebar UI
    # ==========================================================
    st.sidebar.title("⚙️ Налаштування")

    st.session_state.online_mode = st.sidebar.toggle(
        "Online mode (дозволити інтернет-запити)",
        value=bool(st.session_state.online_mode),
        help="Online = sync та LLM. Offline = тільки локальна база.",
    )
    _online_badge()

    # -----------------------------
    # Upload ingest
    # -----------------------------
    with st.sidebar.expander("📄 Upload PDF/DOCX (Local ingest)", expanded=False):
        uploaded_files = st.file_uploader(
            "📎 Завантаж PDF/DOCX файли",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("📥 Ingest uploaded files", use_container_width=True):
            try:
                import importlib

                mod = importlib.import_module("core.upload_ingest")
                ingest_uploaded_files = getattr(mod, "ingest_uploaded_files")
            except Exception as e:
                st.error("❌ Не вдалося імпортувати core/upload_ingest.py")
                st.exception(e)
            else:
                with st.spinner("Імпортую файли..."):
                    rep = ingest_uploaded_files(uploaded_files)

                st.success("✅ Файли додано у базу та індекс оновлено.")
                st.json(rep)
                _maybe_load_index()

    # -----------------------------
    # Document scope for local uploads
    # -----------------------------
    with st.sidebar.expander("📌 Document scope (local uploads)", expanded=False):
        doc_options: List[Tuple[str, str]] = []

        try:
            meta_path = Path(CONFIG.faiss_meta_path)
            if meta_path.exists():
                meta_chunks = load_chunks_from_jsonl(meta_path)

                seen = set()
                for ch in meta_chunks:
                    if (ch.source_type or "").lower() != "local":
                        continue

                    extra = ch.extra or {}
                    doc_id = extra.get("doc_id")
                    if not doc_id:
                        continue
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)

                    label = extra.get("file_name") or extra.get("saved_as") or doc_id
                    doc_options.append((label, doc_id))
        except Exception:
            doc_options = []

        st.session_state.doc_scope_enabled = st.checkbox(
            "Відповідати тільки по завантаженим файлам",
            value=bool(st.session_state.doc_scope_enabled),
        )

        chosen = st.multiselect(
            "Обрати конкретні файли",
            options=[x[1] for x in doc_options],
            format_func=lambda did: next((lbl for lbl, _id in doc_options if _id == did), did),
            disabled=not st.session_state.doc_scope_enabled,
        )
        st.session_state.doc_scope_ids = set(chosen)

    # -----------------------------
    # Retrieval tuning
    # -----------------------------
    with st.sidebar.expander("🧲 Retrieval tuning (FAISS)", expanded=False):
        st.session_state.min_score = st.slider(
            "min_score",
            0.0,
            1.0,
            float(st.session_state.min_score),
            0.01,
        )
        st.session_state.keyword_filter = st.checkbox("keyword_filter", value=bool(st.session_state.keyword_filter))
        st.session_state.use_structured_extraction = st.checkbox(
            "Structured extraction (entity-level)",
            value=bool(st.session_state.use_structured_extraction),
            help="ON: парсинг сутностей та ролей. OFF: fallback на LLM/summary.",
        )
        st.session_state.use_institute_filter = st.checkbox(
            "Institute-aware filtering",
            value=bool(st.session_state.use_institute_filter),
            help="Фільтрує chunks по інституту з запиту до ранжування.",
        )
        st.session_state.use_semantic_rules = st.checkbox(
            "Semantic intent rules",
            value=bool(st.session_state.use_semantic_rules),
            help="Жорсткі правила для rector/director та рівня сутностей.",
        )
        st.session_state.show_retrieval_debug = st.checkbox(
            "Показати retrieval debug",
            value=bool(st.session_state.show_retrieval_debug),
        )

    # -----------------------------
    # Source filters
    # -----------------------------
    with st.sidebar.expander("🧩 Filters (source types)", expanded=False):
        st.caption("Фільтрує джерела перед rerank/answer.")
        st.session_state.filter_lpnu = st.checkbox("LPNU", value=bool(st.session_state.filter_lpnu))
        st.session_state.filter_tg = st.checkbox("Telegram", value=bool(st.session_state.filter_tg))
        st.session_state.filter_vns = st.checkbox("VNS", value=bool(st.session_state.filter_vns))
        st.session_state.filter_local = st.checkbox("Local", value=bool(st.session_state.filter_local))

    # -----------------------------
    # Answer UI
    # -----------------------------
    with st.sidebar.expander("🧾 Answer UI", expanded=False):
        st.session_state.show_used_sources_only = st.checkbox(
            "Показувати тільки джерела, які використала відповідь",
            value=bool(st.session_state.show_used_sources_only),
            help="Працює найкраще з LLM, бо є цитати [1],[2]...",
        )
        st.session_state.show_chunk_preview = st.checkbox(
            "Показувати прев’ю chunk’ів",
            value=bool(st.session_state.show_chunk_preview),
        )

    # -----------------------------
    # Reranker
    # -----------------------------
    with st.sidebar.expander("🎯 Reranker (покращення Top-K)", expanded=False):
        if not RERANK_AVAILABLE:
            st.warning("Reranker модуль не знайдено (`core/rerank.py`). Функція недоступна.")

        st.session_state.use_reranker_ui = st.checkbox(
            "Увімкнути Reranker",
            value=bool(st.session_state.use_reranker_ui),
            disabled=not RERANK_AVAILABLE,
            help="Переранжує топ-N кандидатів для кращої точності.",
        )

        st.session_state.reranker_model_ui = st.text_input(
            "Reranker model",
            value=st.session_state.reranker_model_ui,
            disabled=not (RERANK_AVAILABLE and st.session_state.use_reranker_ui),
            help="Напр.: cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        st.session_state.reranker_top_n_ui = st.slider(
            "Reranker candidates (top-N)",
            min_value=10,
            max_value=100,
            value=int(st.session_state.reranker_top_n_ui),
            step=5,
            disabled=not (RERANK_AVAILABLE and st.session_state.use_reranker_ui),
        )

        st.caption("ℹ️ Reranker працює повільніше, але дає помітно кращі топ-результати.")

    # -----------------------------
    # VNS (UI only)
    # -----------------------------
    with st.sidebar.expander("🔐 ВНС (опційно, без збереження)", expanded=False):
        st.caption(
            "Логін і пароль зберігаються тільки в оперативній памʼяті (session_state). "
            "Не записуються у файли."
        )
        st.session_state.vns_login = st.text_input("VNS login", value=st.session_state.vns_login)
        st.session_state.vns_password = st.text_input("VNS password", value=st.session_state.vns_password,
                                                      type="password")

        st.write("**Зараз збережено в сесії:**")
        st.write(f"Login: `{st.session_state.vns_login}`")
        st.write(f"Password: `{mask_secret(st.session_state.vns_password)}`")

        if st.button("🧹 Очистити VNS креденшали", use_container_width=True):
            st.session_state.vns_login = ""
            st.session_state.vns_password = ""
            st.success("Креденшали очищено.")

    # -----------------------------
    # Telegram (Auth + Test)
    # -----------------------------
    with st.sidebar.expander("📡 Telegram інтеграція (Auth + Sync)", expanded=False):
        st.caption(
            "Telethon потребує **один раз** авторизувати сесію. "
            "Після цього ingest/sync працює без телефону та коду.\n\n"
            "Сесія зберігається локально у файлі: `data/tg_session.session` (не комітити в Git)."
        )

        st.session_state.tg_api_id = st.text_input("Telegram API ID", value=st.session_state.tg_api_id)
        st.session_state.tg_api_hash = st.text_input("Telegram API HASH", value=st.session_state.tg_api_hash,
                                                     type="password")
        st.session_state.tg_channels = st.text_area("Telegram channels (one per line)",
                                                    value=st.session_state.tg_channels)

        api_id_int = _safe_int(st.session_state.tg_api_id.strip()) if st.session_state.tg_api_id.strip() else None
        api_hash_str = st.session_state.tg_api_hash.strip() if st.session_state.tg_api_hash.strip() else None

        st.divider()
        st.subheader("🔐 Telegram авторизація (1 раз)")

        st.session_state.tg_phone = st.text_input("Телефон (+380...)", value=st.session_state.tg_phone)
        st.session_state.tg_code = st.text_input("Код з Telegram", value=st.session_state.tg_code)
        st.session_state.tg_2fa = st.text_input("2FA пароль (якщо увімкнено)", value=st.session_state.tg_2fa,
                                                type="password")

        colA, colB = st.columns(2)
        send_code_btn = colA.button("📨 Надіслати код", disabled=not st.session_state.online_mode)
        sign_in_btn = colB.button("✅ Підтвердити код", disabled=not st.session_state.online_mode)

        if send_code_btn:
            if not api_id_int or not api_hash_str or not st.session_state.tg_phone.strip():
                st.error("Вкажи api_id, api_hash і номер телефону.")
            else:
                try:
                    from telethon import TelegramClient


                    async def _send_code():
                        async with TelegramClient("data/tg_session", api_id_int, api_hash_str) as client:
                            await client.send_code_request(st.session_state.tg_phone.strip())
                            return True


                    with st.spinner("Надсилаю код..."):
                        _run_async(_send_code(), timeout_sec=25)
                    st.success("✅ Код надіслано. Введи код та натисни 'Підтвердити код'.")
                except Exception as e:
                    st.error(f"❌ Не вдалося надіслати код: {e}")

        if sign_in_btn:
            if not api_id_int or not api_hash_str or not st.session_state.tg_phone.strip() or not st.session_state.tg_code.strip():
                st.error("Вкажи api_id, api_hash, телефон і код.")
            else:
                try:
                    from telethon import TelegramClient
                    from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError


                    async def _sign_in():
                        async with TelegramClient("data/tg_session", api_id_int, api_hash_str) as client:
                            try:
                                await client.sign_in(phone=st.session_state.tg_phone.strip(),
                                                     code=st.session_state.tg_code.strip())
                                return {"ok": True, "msg": "✅ Авторизація успішна! Сесія збережена."}
                            except SessionPasswordNeededError:
                                if not st.session_state.tg_2fa.strip():
                                    return {"ok": False, "msg": "⚠️ Увімкнено 2FA. Введи пароль і повтори."}
                                await client.sign_in(password=st.session_state.tg_2fa.strip())
                                return {"ok": True, "msg": "✅ Авторизація успішна (2FA). Сесія збережена."}
                            except PhoneCodeInvalidError:
                                return {"ok": False, "msg": "❌ Невірний код."}


                    with st.spinner("Авторизація..."):
                        out = _run_async(_sign_in(), timeout_sec=35)
                    if out["ok"]:
                        st.success(out["msg"])
                    else:
                        st.warning(out["msg"])
                except Exception as e:
                    st.error(f"❌ Авторизація не вдалася: {e}")

        st.divider()
        st.subheader("🧪 Test Telegram (last 3 msgs)")

        test_btn = st.button("🧪 Test Telegram (show last 3 msgs)", disabled=not st.session_state.online_mode)

        if test_btn:
            channels = _parse_tg_channels(st.session_state.tg_channels)
            if not api_id_int or not api_hash_str or not channels:
                st.error("Введи api_id, api_hash і хоча б 1 канал (наприклад pbikni).")
            else:
                try:
                    from telethon import TelegramClient


                    async def _test_channel():
                        ch = _normalize_channel(channels[0])
                        async with TelegramClient("data/tg_session", api_id_int, api_hash_str) as client:
                            is_auth = await client.is_user_authorized()
                            if not is_auth:
                                return {"ok": False, "err": "❌ Сесія НЕ авторизована. Спочатку зроби авторизацію вище."}

                            entity = await client.get_entity(ch)
                            msgs = []
                            async for m in client.iter_messages(entity, limit=3):
                                txt = (getattr(m, "message", None) or "").strip()
                                if not txt:
                                    continue
                                msgs.append((m.id, m.date, txt))
                            return {"ok": True, "channel": ch, "msgs": msgs}


                    with st.spinner("Тестую Telegram (до 25 сек)..."):
                        out = _run_async(_test_channel(), timeout_sec=25)

                    if not out["ok"]:
                        st.error(out["err"])
                    else:
                        st.success(f"✅ Канал доступний: {out['channel']}")
                        if not out["msgs"]:
                            st.info("Немає текстових повідомлень у останніх 3 або вони порожні.")
                        for mid, dt, txt in out["msgs"]:
                            st.write(f"**{mid}** • {dt}  \n{txt}")

                except Exception as e:
                    st.error(f"Telegram test failed: {e}")

    # -----------------------------
    # LLM settings
    # -----------------------------
    with st.sidebar.expander("🤖 LLM інтеграція (опційно)", expanded=False):
        st.caption(
            "RAG + LLM генерація. Працює через OpenAI-compatible API: OpenAI / Groq / OpenRouter / Ollama / Custom.\n"
            "🔒 Ключ зберігається лише в session_state."
        )

        st.session_state.llm_enabled = st.checkbox(
            "Увімкнути LLM",
            value=bool(st.session_state.llm_enabled),
            disabled=not st.session_state.online_mode,
            help="Потрібен Online mode.",
        )

        provider = st.selectbox(
            "Провайдер",
            options=PROVIDERS,
            index=PROVIDERS.index(st.session_state.llm_provider),
            disabled=not st.session_state.online_mode,
        )

        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.session_state.llm_model = _provider_default_model(provider)

        _ensure_valid_model_for_provider(st.session_state.llm_provider)

        preset_models = MODEL_PRESETS.get(st.session_state.llm_provider, [])
        st.session_state.use_custom_model = st.checkbox(
            "Вказати модель вручну",
            value=bool(st.session_state.use_custom_model),
            disabled=not st.session_state.online_mode,
        )

        if (not st.session_state.use_custom_model) and preset_models:
            options = preset_models[:]
            if st.session_state.llm_model and st.session_state.llm_model not in options:
                options = [st.session_state.llm_model] + options

            selected_model = st.selectbox(
                "Модель (вибери зі списку)",
                options=options,
                index=options.index(st.session_state.llm_model) if st.session_state.llm_model in options else 0,
                disabled=not st.session_state.online_mode,
            )
            st.session_state.llm_model = selected_model
        else:
            st.session_state.llm_model = st.text_input(
                "Модель (вручну)",
                value=st.session_state.llm_model,
                disabled=not st.session_state.online_mode,
            )

        st.session_state.llm_api_key = st.text_input(
            "API Key",
            value=st.session_state.llm_api_key,
            type="password",
            disabled=(not st.session_state.online_mode) or (st.session_state.llm_provider == "ollama"),
        )

        st.session_state.llm_base_url = st.text_input(
            "Custom Base URL (тільки для custom)",
            value=st.session_state.llm_base_url,
            disabled=(not st.session_state.online_mode) or (st.session_state.llm_provider != "custom"),
            help="Напр.: https://your-openai-compatible-endpoint/v1",
        )

        st.session_state.llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.llm_temperature),
            step=0.05,
            disabled=not st.session_state.online_mode,
        )

        st.session_state.llm_debug = st.checkbox(
            "Показати debug (URL + налаштування без ключа)",
            value=bool(st.session_state.llm_debug),
            disabled=not st.session_state.online_mode,
        )

        if st.session_state.llm_debug:
            base_url = build_base_url(st.session_state.llm_provider, st.session_state.llm_base_url or None)
            st.code(f"Request URL: {base_url}/chat/completions", language="text")

        if st.button("🧪 Test LLM", disabled=not (st.session_state.online_mode and st.session_state.llm_enabled)):
            llm = _build_llm_settings()
            try:
                test_out = chat_completion(
                    llm,
                    messages=[
                        {"role": "system", "content": "Ти тестовий помічник."},
                        {"role": "user", "content": "Напиши 'OK' і поточну дату у форматі YYYY-MM-DD."},
                    ],
                )
                st.success("✅ LLM працює!")
                st.write(test_out)
            except Exception as e:
                st.error(f"LLM test failed: {e}")

    # ==========================================================
    # Index actions
    # ==========================================================
    st.sidebar.divider()

    if st.sidebar.button("📦 Перевірити/завантажити локальний індекс"):
        try:
            load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
            st.session_state.index_ready = True
            st.sidebar.success("FAISS індекс завантажено ✅")
        except Exception as e:
            st.session_state.index_ready = False
            st.sidebar.error(f"Не знайдено індекс: {e}")

    if st.sidebar.button("🛠️ Побудувати індекс з local_cache.jsonl"):
        chunks = load_chunks_from_jsonl(CONFIG.local_cache_path)
        if not chunks:
            st.sidebar.error(
                "local_cache.jsonl порожній або не існує. "
                "Натисни 'Sync knowledge base' або додай документи локально."
            )
        else:
            try:
                build_faiss_index(
                    chunks=chunks,
                    embed_model_name=CONFIG.embed_model_name,
                    index_path=CONFIG.faiss_index_path,
                    meta_path=CONFIG.faiss_meta_path,
                )
                st.session_state.index_ready = True
                st.sidebar.success("Індекс успішно побудовано ✅")
            except Exception as e:
                st.sidebar.error(f"Помилка побудови індексу: {e}")

    # ==========================================================
    # Sync knowledge base
    # ==========================================================
    if st.sidebar.button("🔄 Sync knowledge base (LPNU + TG + rebuild index)",
                         disabled=not st.session_state.online_mode):
        from pipelines.sync_all import sync_all

        channels = _parse_tg_channels(st.session_state.tg_channels)
        api_id = _safe_int(st.session_state.tg_api_id.strip()) if st.session_state.tg_api_id.strip() else None
        api_hash = st.session_state.tg_api_hash.strip() if st.session_state.tg_api_hash.strip() else None

        with st.spinner("Синхронізація знань..."):
            report = sync_all(
                api_id=api_id,
                api_hash=api_hash,
                channels=channels if (api_id and api_hash and channels) else None,
            )

        st.session_state.last_sync_report = report
        st.sidebar.success("Sync завершено ✅")
        st.sidebar.json(report)
        _maybe_load_index()

    # ==========================================================
    # Advanced wipe
    # ==========================================================
    st.sidebar.divider()
    with st.sidebar.expander("🧨 Advanced: wipe local storage", expanded=False):
        st.caption("Видаляє local_cache.jsonl + FAISS index. Корисно для чистих тестів.")
        confirm = st.checkbox("Я розумію що це видалить локальні дані")
        if st.button("🗑️ Wipe local cache + index", disabled=not confirm):
            r = _wipe_local_storage()
            st.session_state.index_ready = False
            st.session_state.last_results = []
            st.session_state.last_sync_report = None
            if r.get("ok"):
                st.success("✅ Видалено. Тепер можна зробити Sync з нуля.")
            else:
                st.error("❌ Не вдалося видалити всі файли (можуть бути відкриті).")

    # ==========================================================
    # UI reset
    # ==========================================================
    st.sidebar.divider()
    if st.sidebar.button("🧹 Очистити результати пошуку (UI)"):
        st.session_state.last_results = []
        st.session_state.last_answer_struct = None
        st.success("Результати очищено.")

    if st.sidebar.button("🧨 Повний скидання (очистити UI + креденшали)"):
        st.session_state.last_results = []
        st.session_state.last_answer_struct = None
        st.session_state.vns_login = ""
        st.session_state.vns_password = ""
        st.session_state.tg_api_id = ""
        st.session_state.tg_api_hash = ""
        st.session_state.tg_phone = ""
        st.session_state.tg_code = ""
        st.session_state.tg_2fa = ""
        st.session_state.llm_api_key = ""
        st.session_state.last_sync_report = None
        st.session_state.last_eval_metrics = None
        st.session_state.last_eval_df = None
        st.session_state.last_eval_plots_dir = None
        st.success("Сесія очищена (без видалення файлів).")

    # ==========================================================
    # Main UI
    # ==========================================================
    st.title("🎓 RAG Assistant — RAG MVP (Streamlit)")

    st.caption(
        "MVP: локальний індекс + retrieval + відповідь (offline або через LLM). "
        "Є авто-архів даних (LPNU + Wiki + Telegram) через 'Sync knowledge base'."
    )

    if st.session_state.online_mode:
        st.info("🟢 ONLINE режим увімкнено — можна синхронізувати дані та використовувати LLM.")
    else:
        st.warning("🔴 OFFLINE режим — працює тільки локальна база та індекс.")

    if not Path(CONFIG.faiss_index_path).exists():
        st.warning(
            "FAISS індекс ще не створено. Натисни **Sync knowledge base** у сайдбарі (Online mode), "
            "щоб завантажити всі інститути ЛПНУ та побудувати індекс."
        )

    if Path(CONFIG.faiss_index_path).exists() and not st.session_state.index_ready:
        _maybe_load_index()

    if st.session_state.last_sync_report:
        with st.expander("📄 Останній Sync report", expanded=False):
            st.json(st.session_state.last_sync_report)

    tab_chat, tab_eval, tab_research = st.tabs(["💬 Chat / Search", "📊 Metrics & Evaluation", "🧪 Research & Ops"])

    # ==========================================================
    # TAB 1: Chat / Search
    # ==========================================================
    with tab_chat:
        st.subheader("⚡ Швидкі запити (усі інститути ЛПНУ)")

        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        if qcol1.button("Керівництво інституту"):
            st.session_state.quick_query = "Хто входить до керівництва інституту та які їх посади?"
        if qcol2.button("Історія інституту"):
            st.session_state.quick_query = "Коли було засновано інститут і які ключові події його історії?"
        if qcol3.button("Освітні програми"):
            st.session_state.quick_query = "Які бакалаврські та магістерські програми пропонує інститут?"
        if qcol4.button("Новини університету"):
            st.session_state.quick_query = "Останні новини та оголошення інституту"

        qcol5, qcol6, qcol7, qcol8 = st.columns(4)
        if qcol5.button("Контакти інституту"):
            st.session_state.quick_query = "Контактна інформація викладачів та адміністрації інституту"
        if qcol6.button("Розклад подій"):
            st.session_state.quick_query = "Розклад лекцій, семінарів та академічних заходів інституту"
        if qcol7.button("Гуртожитки"):
            st.session_state.quick_query = "Інформація про гуртожитки, правила поселення та студентське життя"
        if qcol8.button("Бібліотека та ресурси"):
            st.session_state.quick_query = "Електронні ресурси, правила користування бібліотекою та доступ до навчальних матеріалів"

        default_query = st.session_state.get("quick_query", "")

        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            st.subheader("🔎 Запит")

            query = st.text_input("Введи питання", value=default_query)
            top_k = st.slider("Top-K джерел", min_value=1, max_value=10, value=int(CONFIG.top_k))

            use_llm = bool(st.session_state.online_mode and st.session_state.llm_enabled)

            if use_llm:
                st.caption(
                    f"🤖 Генерація: **LLM ON** • provider=`{st.session_state.llm_provider}` • model=`{st.session_state.llm_model}`"
                )
            else:
                st.caption("📌 Генерація: **LLM OFF** (offline summarizer)")

            if st.session_state.use_reranker_ui and RERANK_AVAILABLE:
                st.caption(
                    f"🎯 Reranker: **ON** • model=`{st.session_state.reranker_model_ui}` • topN={st.session_state.reranker_top_n_ui}"
                )
            else:
                st.caption("🎯 Reranker: **OFF**")
            st.caption(f"🧩 Extraction: **{'ON' if st.session_state.use_structured_extraction else 'OFF'}**")
            st.caption(f"🏛️ Institute filter: **{'ON' if st.session_state.use_institute_filter else 'OFF'}**")
            st.caption(f"🛡️ Rules: **{'ON' if st.session_state.use_semantic_rules else 'OFF'}**")

            ask_btn = st.button("Отримати відповідь", type="primary", use_container_width=True)

            if ask_btn:
                st.session_state.quick_query = ""
                st.session_state.last_query = query

                if not query.strip():
                    st.warning("Введи запит.")
                elif not st.session_state.index_ready:
                    st.warning("Спершу завантаж або побудуй локальний FAISS індекс у сайдбарі.")
                else:
                    index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)

                    llm = _build_llm_settings() if use_llm else None
                    query_config = _build_current_query_config(top_k=top_k, llm=llm)
                    ans, bundle = answer_query_with_config(
                        query=query,
                        index=index,
                        chunks=meta,
                        config=query_config,
                    )

                    results = bundle.context_results or bundle.results
                    st.session_state.last_results = results
                    ranking_mode = "adaptive_hybrid_reranker" if (st.session_state.use_reranker_ui and RERANK_AVAILABLE) else "adaptive_hybrid"

                    st.session_state.last_answer_struct = ans

                    st.markdown(ans.markdown, unsafe_allow_html=True)

                    if getattr(ans, "warnings", None):
                        st.warning(" | ".join(ans.warnings))

                    st.divider()
                    st.markdown("### 👍👎 Feedback")

                    comment = st.text_input("Коментар (опційно)", key="feedback_comment")

                    c1, c2, _ = st.columns([1, 1, 3])
                    good = c1.button("👍 Добре", use_container_width=True)
                    bad = c2.button("👎 Погано", use_container_width=True)

                    if good or bad:
                        rating = +1 if good else -1
                        payload = _feedback_payload(rating=rating, comment=st.session_state.get("feedback_comment", ""))
                        _append_feedback(payload)
                        st.success("✅ Дякую! Фідбек збережено.")
                        st.session_state.feedback_comment = ""

                    if st.session_state.show_retrieval_debug:
                        with st.expander("🧪 Retrieval debug", expanded=False):
                            st.write(
                                {
                                    "question_type": bundle.analysis.question_type,
                                    "expanded_query": bundle.analysis.expanded_query,
                                    "adaptive_top_k": bundle.analysis.adaptive_top_k,
                                    "candidates_after_search": len(bundle.candidates),
                                    "context_results": len(bundle.context_results),
                                    "top_k_returned": len(results),
                                    "ranking_mode": ranking_mode,
                                    "min_score": float(st.session_state.min_score),
                                    "keyword_filter": bool(st.session_state.keyword_filter),
                                    "allowed_types": sorted(list(query_config.allowed_types or [])),
                                    "doc_scope_enabled": bool(st.session_state.doc_scope_enabled),
                                    "doc_scope_ids": sorted(list(st.session_state.doc_scope_ids or [])),
                                    "use_structured_extraction": bool(st.session_state.use_structured_extraction),
                                    "use_institute_filter": bool(st.session_state.use_institute_filter),
                                    "use_rules": bool(st.session_state.use_semantic_rules),
                                    "query_institute": bundle.analysis.institute_code,
                                    "detected_intent": bundle.analysis.intent,
                                    "entity_scope": bundle.analysis.entity_scope,
                                    "used_sources": getattr(ans, "used_sources", []),
                                    "warnings": getattr(ans, "warnings", []),
                                }
                            )
                            st.markdown("**Query**")
                            st.code(query, language="text")
                            if bundle.entity_candidates:
                                st.markdown("**Extracted candidates (entity-level)**")
                                st.dataframe(pd.DataFrame(bundle.entity_candidates), use_container_width=True, hide_index=True)
                            else:
                                st.info("No entity candidates extracted for this query.")
                            if getattr(bundle, "accepted_candidates", None):
                                st.markdown("**Accepted candidates**")
                                st.dataframe(pd.DataFrame(bundle.accepted_candidates), use_container_width=True, hide_index=True)
                            if getattr(bundle, "rejected_candidates", None):
                                st.markdown("**Rejected candidates (with reasons)**")
                                st.dataframe(pd.DataFrame(bundle.rejected_candidates), use_container_width=True, hide_index=True)
                            st.markdown("**Retrieved chunks**")
                            debug_rows = []
                            for rank, (chunk, score) in enumerate(results, start=1):
                                debug_rows.append(
                                    {
                                        "rank": rank,
                                        "score": round(float(score), 4),
                                        "title": chunk.title,
                                        "type": chunk.source_type,
                                        "url": chunk.url,
                                        "snippet": (chunk.text[:160] + "...") if len(chunk.text) > 160 else chunk.text,
                                    }
                                )
                            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
                            st.markdown("**Selected answer**")
                            st.code(ans.answer_text or "", language="text")
                            st.markdown("**Confidence**")
                            st.code(ans.confidence, language="text")
                            st.markdown("**Final decision**")
                            st.code(getattr(bundle, "final_decision", ""), language="text")

        with col2:
            st.subheader("📚 Джерела")

            if not st.session_state.last_results:
                st.info("Після запиту тут зʼявляться знайдені фрагменти (top-K).")
            else:
                ans = st.session_state.last_answer_struct
                used = set(getattr(ans, "used_sources", []) or []) if ans else set()

                rows = []
                for rank, (chunk, score) in enumerate(st.session_state.last_results, start=1):
                    if st.session_state.show_used_sources_only and used and (rank not in used):
                        continue

                    rows.append(
                        {
                            "Rank": rank,
                            "Used": "✅" if (rank in used) else "",
                            "Score": round(float(score), 4),
                            "Institute": ((chunk.extra or {}).get("institute_code") or "").upper() or "N/A",
                            "Title": chunk.title,
                            "Type": chunk.source_type,
                            "Date": chunk.date,
                            "URL": chunk.url,
                            "DocID": (chunk.extra or {}).get("doc_id"),
                            "Text (snippet)": (chunk.text[:180] + "…") if len(chunk.text) > 180 else chunk.text,
                        }
                    )

                df = pd.DataFrame(rows)
                if not df.empty:
                    grouped = df.sort_values(["Institute", "Rank"]).groupby("Institute", sort=False)
                    for institute, gdf in grouped:
                        st.markdown(f"**{institute}**")
                        st.dataframe(gdf.head(2), use_container_width=True, hide_index=True)
                else:
                    st.info("Немає джерел для показу.")

                if st.session_state.show_chunk_preview:
                    st.markdown("#### 🔍 Повний текст chunk’ів")
                    for rank, (chunk, score) in enumerate(st.session_state.last_results, start=1):
                        if st.session_state.show_used_sources_only and used and (rank not in used):
                            continue

                        label = f"[{rank}] {chunk.title} ({chunk.source_type}) score={score:.3f}"
                        with st.expander(label, expanded=False):
                            if chunk.url:
                                st.write(chunk.url)
                            if chunk.date:
                                st.write(chunk.date)

                            extra = chunk.extra or {}
                            if extra.get("doc_id"):
                                st.caption(f"doc_id: `{extra.get('doc_id')}`")

                            st.write(chunk.text)

else:
    st.sidebar.title("⚡ Простий режим")

    st.session_state.online_mode = st.sidebar.toggle(
        "Online",
        value=bool(st.session_state.online_mode),
    )

    st.session_state.llm_enabled = True

    # дефолти щоб система працювала стабільно
    st.session_state.min_score = 0.2
    st.session_state.keyword_filter = False
    st.session_state.use_reranker_ui = True
    st.session_state.reranker_top_n_ui = 20

    st.sidebar.caption("Все працює автоматично 👌")

if mode == "LIGHT":
    st.title("💬 RAG Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # показ історії
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Напиши питання...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):

                index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)
                llm = _build_llm_settings() if st.session_state.online_mode else None
                light_config = AnswerQueryConfig(
                    embed_model_name=CONFIG.embed_model_name,
                    min_score=0.2,
                    keyword_filter=False,
                    allowed_types=("lpnu", "tg", "vns", "local"),
                    use_reranker=bool(RERANK_AVAILABLE),
                    reranker_model=st.session_state.reranker_model_ui,
                    reranker_top_n=20,
                    use_query_expansion=True,
                    use_adaptive_top_k=True,
                    use_hybrid=True,
                    top_k_override=5,
                    use_post_boosts=True,
                    use_extraction=bool(st.session_state.use_structured_extraction),
                    use_institute_filter=bool(st.session_state.use_institute_filter),
                    use_rules=bool(st.session_state.use_semantic_rules),
                    llm=llm,
                )
                ans, bundle = answer_query_with_config(
                    query=prompt,
                    index=index,
                    chunks=meta,
                    config=light_config,
                )

                st.markdown(ans.markdown)

        st.session_state.messages.append({
            "role": "assistant",
            "content": ans.markdown
        })

    st.stop()  # 🚨 ЗУПИНЯЄ виконання, щоб PRO UI не рендерився


# ==========================================================
# TAB 2: Metrics & Evaluation
# ==========================================================
with tab_eval:
    st.subheader("📊 Metrics & Evaluation")
    st.write("Оцінка retrieval (Recall/MRR/Precision) на eval_set.jsonl + графіки для звіту.")

    eval_k = st.slider("Evaluation K (top-K)", min_value=1, max_value=10, value=5)

    use_reranker_eval = st.checkbox(
        "Використати Reranker під час evaluation",
        value=bool(st.session_state.use_reranker_ui and RERANK_AVAILABLE),
        disabled=not RERANK_AVAILABLE,
    )

    if st.button("🚀 Run evaluation", type="primary"):
        if not st.session_state.index_ready:
            st.error("Спершу завантаж/побудуй індекс.")
        elif not Path("eval_set.jsonl").exists():
            st.error("Файл `eval_set.jsonl` не знайдено.")
        else:
            with st.spinner("Оцінювання retrieval..."):
                out = run_retrieval_eval(top_k=eval_k, use_reranker=use_reranker_eval)

            if out.get("ok"):
                st.session_state.last_eval_metrics = out["metrics"]
                st.session_state.last_eval_df = out["df"]
                st.session_state.last_eval_plots_dir = out["plots_dir"]
                st.success("✅ Готово! Результати збережено у `report/`.")
            else:
                st.error(out.get("error", "Evaluation failed."))

    if st.button("⚖️ Compare without/with institute filter"):
        if not st.session_state.index_ready:
            st.error("Спершу завантаж/побудуй індекс.")
        elif not Path("eval_set.jsonl").exists():
            st.error("Файл `eval_set.jsonl` не знайдено.")
        else:
            with st.spinner("Запускаю два прогони evaluation..."):
                out_off = run_retrieval_eval(top_k=eval_k, use_reranker=use_reranker_eval, use_institute_filter=False)
                out_on = run_retrieval_eval(top_k=eval_k, use_reranker=use_reranker_eval, use_institute_filter=True)
            if out_off.get("ok") and out_on.get("ok"):
                compare_df = pd.DataFrame(
                    [
                        {
                            "mode": "without_institute_filter",
                            "Recall@K": out_off["metrics"].get("recall_at_k", 0.0),
                            "Hit@1": out_off["metrics"].get("hit_at_1", 0.0),
                            "Exact Match": out_off["metrics"].get("exact_match", 0.0),
                            "F1": out_off["metrics"].get("f1", 0.0),
                        },
                        {
                            "mode": "with_institute_filter",
                            "Recall@K": out_on["metrics"].get("recall_at_k", 0.0),
                            "Hit@1": out_on["metrics"].get("hit_at_1", 0.0),
                            "Exact Match": out_on["metrics"].get("exact_match", 0.0),
                            "F1": out_on["metrics"].get("f1", 0.0),
                        },
                    ]
                )
                st.markdown("### Institute-aware impact")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

    if st.session_state.last_eval_metrics:
        m = st.session_state.last_eval_metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall@K", f"{m['recall_at_k']:.3f}")
        c2.metric("MRR@K", f"{m['mrr_at_k']:.3f}")
        c3.metric("Avg Precision@K", f"{m['avg_precision_at_k']:.3f}")
        c4.metric("Mode", str(m.get("mode", "faiss")))

        st.caption(
            f"Embedding model: `{m.get('embed_model')}` | "
            f"min_score={m.get('min_score')} | keyword_filter={m.get('keyword_filter')}"
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Hit@1", f"{m.get('hit_at_1', 0.0):.3f}")
        c6.metric("Hit@3", f"{m.get('hit_at_3', 0.0):.3f}")
        c7.metric("Hit@5", f"{m.get('hit_at_5', 0.0):.3f}")
        c8.metric("Top1 mean", f"{m.get('top1_score_mean'):.3f}" if m.get("top1_score_mean") is not None else "—")

        c8b, c8c, c8d = st.columns(3)
        c8b.metric("MAP@K", f"{m.get('map_at_k', 0.0):.3f}")
        c8c.metric("nDCG@K", f"{m.get('ndcg_at_k', 0.0):.3f}")
        c8d.metric("Grounding", f"{m.get('grounding_score'):.3f}" if m.get("grounding_score") is not None else "N/A")

        c9, c10, c10b = st.columns(3)
        exact_match_value = m.get("exact_match")
        f1_value = m.get("f1")
        c9.metric("Exact Match", f"{exact_match_value:.3f}" if exact_match_value is not None else "N/A")
        c10.metric("F1", f"{f1_value:.3f}" if f1_value is not None else "N/A")
        c10b.metric("Semantic F1", f"{m.get('semantic_f1'):.3f}" if m.get("semantic_f1") is not None else "N/A")
        c11, c12, c13, c14 = st.columns(4)
        c11.metric("Invalid role", str(m.get("invalid_role_answers", 0)))
        c12.metric("Wrong institute", str(m.get("wrong_institute_answers", 0)))
        c13.metric("Hallucinations", str(m.get("hallucinations", 0)))
        c14.metric("Answer refs", f"{m.get('answer_metric_count', 0)}/{m.get('answer_metric_count', 0) + m.get('answer_metric_skipped', 0)}")
        c15, c16, c17, c18 = st.columns(4)
        c15.metric("Source trust", f"{m.get('source_confidence'):.3f}" if m.get("source_confidence") is not None else "N/A")
        c16.metric("Citation ok", f"{m.get('citation_correctness'):.3f}" if m.get("citation_correctness") is not None else "N/A")
        c17.metric("Latency ms", f"{m.get('avg_latency_ms', 0.0):.1f}")
        c18.metric("Est. cost USD", f"{m.get('estimated_cost_usd'):.4f}" if m.get("estimated_cost_usd") is not None else "N/A")

    if isinstance(st.session_state.last_eval_df, pd.DataFrame):
        st.markdown("### 📋 Evaluation table")
        st.dataframe(st.session_state.last_eval_df, use_container_width=True, hide_index=True)

    diploma_summary_path = Path("report/diploma/diploma_summary.csv")
    if diploma_summary_path.exists():
        st.markdown("### 🏆 Diploma leaderboard")
        try:
            diploma_df = pd.read_csv(diploma_summary_path)
            sort_columns = [col for col in ["hit_at_1", "mrr_at_k", "exact_match", "f1"] if col in diploma_df.columns]
            if sort_columns:
                diploma_df = diploma_df.sort_values(sort_columns, ascending=False)
            st.dataframe(diploma_df.head(15), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Не вдалося прочитати leaderboard: {exc}")

    # ==========================================================
    # Button: Test multiple embeddings and parameters (optimized)
    # ==========================================================
    import gc
    import torch


    def clear_memory():
        """Очищає пам’ять CPU і GPU після великої моделі."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    st.divider()
    st.subheader("⚙️ Embedding & Parameter Tuning (Safe Sweep)")
    st.caption("Для коректного порівняння різних embedding-моделей краще запускати `scripts/run_diploma_experiments.py`, бо він перебудовує індекс під кожну модель окремо.")

    if st.button("🧪 Run embedding + parameter sweep (optimized)"):
        # Менші chunk_sizes для безпечного sweep
        embed_models = ["all-MiniLM-L6-v2", "intfloat/e5-small"]  # прибрали LaBSE для тесту
        chunk_sizes = [250, 500]
        top_ks = [5, 10]
        min_scores = [0.3, 0.35]

        results_summary = []

        use_reranker_eval = st.checkbox(
            "Use reranker during sweep",
            value=bool(st.session_state.use_reranker_ui and RERANK_AVAILABLE),
            disabled=not RERANK_AVAILABLE,
        )

        with st.spinner("Running parameter sweep... This may take a while 🚀"):
            for em in embed_models:
                CONFIG.embed_model_name = em
                st.write(f"🔹 Testing embedding model: {em}")

                # Завантажуємо модель один раз на весь цикл chunk_size/top_k/min_score
                for chunk_size in chunk_sizes:
                    CONFIG.chunk_size = chunk_size
                    st.write(f"  • Chunk size: {chunk_size}")

                    for top_k in top_ks:
                        for min_s in min_scores:
                            st.session_state.min_score = min_s
                            st.write(f"    - top_k={top_k}, min_score={min_s}")

                            # Виклик основної функції оцінки
                            out = run_retrieval_eval(top_k=top_k, use_reranker=use_reranker_eval)
                            metrics = out["metrics"]
                            df = out["df"]

                            # Збираємо результати
                            results_summary.append({
                                "embed_model": em,
                                "chunk_size": chunk_size,
                                "top_k": top_k,
                                "min_score": min_s,
                                "recall_at_k": metrics.get("recall_at_k"),
                                "mrr_at_k": metrics.get("mrr_at_k"),
                                "avg_precision": metrics.get("avg_precision_at_k"),
                                "hit_at_1": metrics.get("hit_at_1"),
                                "rouge_l_mean": df["rouge_l"].mean() if "rouge_l" in df else None,
                                "semantic_sim_mean": df["semantic_sim"].mean() if "semantic_sim" in df else None,
                            })

                            # Чистимо пам’ять після кожної ітерації
                            clear_memory()

        # Підсумковий DataFrame
        df_summary = pd.DataFrame(results_summary)
        st.success("✅ Sweep finished!")
        st.subheader("📊 Embedding & Parameter Tuning Results")
        st.dataframe(df_summary.sort_values("recall_at_k", ascending=False), use_container_width=True)

    st.markdown("### 📈 Plots")
    if st.session_state.last_eval_plots_dir:
        plots_dir = Path(st.session_state.last_eval_plots_dir)
        for p in ["hit_ratio.png", "hit_rank_hist.png", "top1_score_hist.png"]:
            fp = plots_dir / p
            if fp.exists():
                st.image(str(fp), caption=p)

    st.divider()
    st.markdown("### 📦 Dataset statistics (local_cache.jsonl)")
    stats = dataset_stats_from_cache(Path(CONFIG.local_cache_path))
    if stats.get("ok"):
        df_types = pd.DataFrame([{"source_type": k, "chunks": v} for k, v in stats["types"].items()])
        st.dataframe(df_types, use_container_width=True, hide_index=True)
        st.bar_chart(df_types.set_index("source_type"))

    st.divider()
    st.markdown("### 📦 Export for report (ZIP)")
    if st.button("📥 Export report package (ZIP)"):
        zp = export_report_zip()
        if not zp:
            st.error("Немає папки report/ або файлів. Спершу запусти evaluation.")
        else:
            with open(zp, "rb") as f:
                st.download_button(
                    label="⬇️ Download report_package.zip",
                    data=f,
                    file_name="report_package.zip",
                    mime="application/zip",
                )

    st.divider()
    st.markdown("## 🚨 Bad queries panel (Feedback)")

    with st.expander("Bad queries", expanded=False):
        if not FEEDBACK_PATH.exists():
            st.info("Ще немає feedback.")
        else:
            rows = safe_read_jsonl(FEEDBACK_PATH, limit=5000)
            if not rows:
                st.info("Feedback файл є, але не вдалося зчитати JSONL.")
            else:
                df_fb = pd.DataFrame(rows)
                st.dataframe(df_fb, use_container_width=True, hide_index=True)

                if "rating" in df_fb.columns:
                    bad_df = df_fb[df_fb["rating"] == -1]
                    st.caption(f"Bad queries: {len(bad_df)}")
                    st.dataframe(bad_df, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 3: Research & Ops
# ==========================================================
with tab_research:
    st.subheader("🧪 Research & Ops")
    st.write("Benchmark suite, expanded evaluation, self-improvement loop, validation та аналіз помилок для дипломного захисту.")

    research_top_k = st.slider("Benchmark Top-K", min_value=1, max_value=10, value=5, key="research_top_k")
    research_min_score = st.slider("Benchmark min_score", min_value=0.0, max_value=0.6, value=float(st.session_state.min_score), step=0.05, key="research_min_score")

    r1, r2, r3, r4 = st.columns(4)
    if r1.button("Run benchmark suite", use_container_width=True):
        if not st.session_state.index_ready:
            st.error("Спершу побудуй або завантаж індекс.")
        else:
            from core.benchmarking import run_benchmark_suite

            llm = _build_llm_settings() if (st.session_state.online_mode and st.session_state.llm_enabled) else None
            with st.spinner("Запускаю benchmark suite..."):
                bench = run_benchmark_suite(
                    embed_model=CONFIG.embed_model_name,
                    reranker_model=st.session_state.reranker_model_ui,
                    top_k=int(research_top_k),
                    min_score=float(research_min_score),
                    llm=llm,
                    allowed_types=_current_allowed_types(),
                )
            st.session_state.last_benchmark_summary = bench["summary"]
            st.session_state.last_benchmark_details = bench["details"]
            st.success(f"Benchmark suite завершено: {bench['run_dir']}")

    if r2.button("Validate KB", use_container_width=True):
        from core.kb_validation import validate_local_knowledge_base

        report = validate_local_knowledge_base(CONFIG.local_cache_path)
        st.session_state.last_kb_validation = report
        st.success("Knowledge base validation завершено.")

    if r3.button("Build large eval set", use_container_width=True):
        from core.eval_dataset import expand_eval_dataset

        report = expand_eval_dataset(output_path=CONFIG.expanded_eval_path)
        st.session_state.last_self_improve_report = {"expanded_eval_only": report}
        st.success(f"Expanded eval set створено: {report['expanded_examples']} прикладів.")

    if r4.button("Self Improve", use_container_width=True):
        from core.self_improve import run_self_improve

        with st.spinner("Збираю hard cases, розширюю eval set та оновлюю індекс..."):
            report = run_self_improve(
                reindex=True,
                expanded_eval_output=CONFIG.expanded_eval_path,
                report_path=CONFIG.self_improve_report_path,
            )
        st.session_state.last_self_improve_report = report
        st.success("Self-improve pipeline завершено.")

    st.divider()
    st.markdown("### Benchmark history")
    try:
        from core.benchmarking import load_benchmark_history

        history_df = load_benchmark_history(CONFIG.benchmark_dir)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("Історія benchmark runs поки порожня.")
    except Exception as exc:
        st.warning(f"Не вдалося прочитати benchmark history: {exc}")

    if isinstance(st.session_state.last_benchmark_summary, pd.DataFrame):
        st.markdown("### Last benchmark leaderboard")
        st.dataframe(st.session_state.last_benchmark_summary, use_container_width=True, hide_index=True)
    if isinstance(st.session_state.last_benchmark_details, pd.DataFrame):
        st.markdown("### Last benchmark failure analysis")
        failure_df = st.session_state.last_benchmark_details
        if "exact_match" in failure_df.columns:
            failure_df = failure_df[failure_df["exact_match"].fillna(0) < 1.0]
        st.dataframe(failure_df.head(100), use_container_width=True, hide_index=True)

    if st.session_state.last_kb_validation:
        st.markdown("### Knowledge base validation")
        st.json(st.session_state.last_kb_validation)

    if st.session_state.last_self_improve_report:
        st.markdown("### Self-improve report")
        st.json(st.session_state.last_self_improve_report)

    eval_errors_path = Path("report/eval_errors.jsonl")
    if eval_errors_path.exists():
        st.markdown("### Failure cases viewer")
        error_rows = safe_read_jsonl(eval_errors_path, limit=200)
        if error_rows:
            st.dataframe(pd.DataFrame(error_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Поки немає помилок для показу.")


# ==========================================================
# System status
# ==========================================================
st.divider()
st.subheader("🧪 Статус системи")
st.write(
    {
        "online_mode": st.session_state.online_mode,
        "index_ready": st.session_state.index_ready,
        "local_cache_path": str(CONFIG.local_cache_path),
        "index_path": str(CONFIG.faiss_index_path),
        "meta_path": str(CONFIG.faiss_meta_path),
        "embed_model": CONFIG.embed_model_name,
        "llm_enabled": bool(st.session_state.llm_enabled),
        "llm_provider": st.session_state.llm_provider,
        "llm_model": st.session_state.llm_model,
        "reranker_available": bool(RERANK_AVAILABLE),
        "reranker_enabled": bool(st.session_state.use_reranker_ui and RERANK_AVAILABLE),
        "doc_scope_enabled": bool(st.session_state.doc_scope_enabled),
        "doc_scope_ids": sorted(list(st.session_state.doc_scope_ids or [])),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
)
