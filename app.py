# app.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Optional

import streamlit as st
import pandas as pd

from core.config import CONFIG
from core.index import load_faiss_index, build_faiss_index, load_chunks_from_jsonl
from core.rag import make_answer_no_llm, make_answer_with_llm
from core.security import mask_secret
from core.llm import LLMSettings, chat_completion, build_base_url


st.set_page_config(page_title=CONFIG.project_name, layout="wide")


# -----------------------------
# Model presets
# -----------------------------
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


# -----------------------------
# Session state init
# -----------------------------
def _init_state():
    defaults = {
        # VNS
        "vns_login": "",
        "vns_password": "",
        # Online mode
        "online_mode": True,
        # RAG
        "last_results": [],
        "index_ready": False,
        "last_sync_report": None,
        # Telegram
        "tg_api_id": "",
        "tg_api_hash": "",
        "tg_channels": "pbikni",
        "tg_phone": "",
        "tg_code": "",
        "tg_2fa": "",
        "tg_auth_status": "unknown",
        # LLM
        "llm_enabled": False,
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "",
        "llm_base_url": "",
        "llm_temperature": 0.2,
        "llm_debug": False,
        # UI
        "quick_query": "",
        "use_custom_model": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# -----------------------------
# Helpers
# -----------------------------
def _parse_tg_channels(raw: str) -> List[str]:
    chans: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        chans.append(line)
    return chans


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


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
        model=st.session_state.llm_model.strip(),
        api_key=st.session_state.llm_api_key.strip(),
        base_url=st.session_state.llm_base_url.strip() or None,
        temperature=float(st.session_state.llm_temperature),
        max_tokens=650,
    )


def _provider_default_model(provider: str) -> str:
    presets = MODEL_PRESETS.get(provider) or []
    return presets[0] if presets else ""


def _ensure_valid_model_for_provider(provider: str) -> None:
    presets = MODEL_PRESETS.get(provider) or []
    if provider == "custom":
        return
    if not presets:
        return
    if st.session_state.llm_model not in presets:
        st.session_state.llm_model = presets[0]


def _delete_file_silent(p: Path) -> bool:
    try:
        if p.exists():
            p.unlink()
        return True
    except Exception:
        return False


def _wipe_local_storage() -> dict:
    """
    Delete local_cache + index files (for clean tests).
    """
    ok = True
    ok &= _delete_file_silent(Path(CONFIG.local_cache_path))
    ok &= _delete_file_silent(Path(CONFIG.faiss_index_path))
    ok &= _delete_file_silent(Path(CONFIG.faiss_meta_path))
    return {"ok": ok}


# -------- Telegram async helpers (SAFE for Streamlit) ----------
def _run_async(coro, timeout_sec: int = 25):
    """
    Runs coroutine with a timeout in a fresh loop.
    Prevents Streamlit from hanging forever.
    """
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


def _normalize_channel(channel: str) -> str:
    ch = (channel or "").strip()
    ch = ch.replace("https://t.me/", "").replace("http://t.me/", "").replace("t.me/", "")
    ch = ch.strip("@").strip("/").strip()
    return ch


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Налаштування")

st.session_state.online_mode = st.sidebar.toggle(
    "Online mode (дозволити інтернет-запити)",
    value=st.session_state.online_mode,
    help="Online = sync та LLM. Offline = тільки локальна база.",
)
_online_badge()

# -----------------------------
# VNS (UI only for now)
# -----------------------------
with st.sidebar.expander("🔐 ВНС (опційно, без збереження)", expanded=False):
    st.caption(
        "Логін і пароль зберігаються тільки в оперативній памʼяті (session_state). "
        "Не записуються у файли."
    )
    st.session_state.vns_login = st.text_input("VNS login", value=st.session_state.vns_login)
    st.session_state.vns_password = st.text_input("VNS password", value=st.session_state.vns_password, type="password")

    st.write("**Зараз збережено в сесії:**")
    st.write(f"Login: `{st.session_state.vns_login}`")
    st.write(f"Password: `{mask_secret(st.session_state.vns_password)}`")

    if st.button("🧹 Очистити VNS креденшали"):
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
    st.session_state.tg_api_hash = st.text_input("Telegram API HASH", value=st.session_state.tg_api_hash, type="password")
    st.session_state.tg_channels = st.text_area("Telegram channels (one per line)", value=st.session_state.tg_channels)

    api_id_int = _safe_int(st.session_state.tg_api_id.strip()) if st.session_state.tg_api_id.strip() else None
    api_hash_str = st.session_state.tg_api_hash.strip() if st.session_state.tg_api_hash.strip() else None

    st.divider()
    st.subheader("🔐 Telegram авторизація (1 раз)")

    st.session_state.tg_phone = st.text_input("Телефон (+380...)", value=st.session_state.tg_phone)
    st.session_state.tg_code = st.text_input("Код з Telegram", value=st.session_state.tg_code)
    st.session_state.tg_2fa = st.text_input("2FA пароль (якщо увімкнено)", value=st.session_state.tg_2fa, type="password")

    colA, colB = st.columns(2)
    send_code_btn = colA.button("📨 Надіслати код", disabled=not st.session_state.online_mode)
    sign_in_btn = colB.button("✅ Підтвердити код", disabled=not st.session_state.online_mode)

    # async actions
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
                            await client.sign_in(phone=st.session_state.tg_phone.strip(), code=st.session_state.tg_code.strip())
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

    # session status
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
        value=st.session_state.llm_enabled,
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

    st.caption(
        "ℹ️ **Temperature впливає на стиль відповіді:**\n"
        "- **0.0 – 0.3** → сухо, коротко, стабільно\n"
        "- **0.4 – 0.7** → баланс, природна мова\n"
        "- **0.8 – 1.0** → довше, творчіше (можливі відхилення)"
    )

    st.session_state.llm_debug = st.checkbox(
        "Показати debug (URL + налаштування без ключа)",
        value=bool(st.session_state.llm_debug),
        disabled=not st.session_state.online_mode,
    )

    if st.session_state.llm_debug:
        base_url = build_base_url(st.session_state.llm_provider, st.session_state.llm_base_url or None)
        st.code(f"Request URL: {base_url}/chat/completions", language="text")
        st.json(
            {
                "enabled": bool(st.session_state.llm_enabled),
                "provider": st.session_state.llm_provider,
                "model": st.session_state.llm_model,
                "temperature": float(st.session_state.llm_temperature),
                "base_url": st.session_state.llm_base_url or "(auto)",
                "api_key": "(set)" if st.session_state.llm_api_key else "(empty)",
            }
        )

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


st.sidebar.divider()

# -----------------------------
# Index actions
# -----------------------------
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
            "Натисни 'Sync knowledge base' щоб завантажити LPNU/TG або додай документи локально."
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


# -----------------------------
# Sync knowledge base
# -----------------------------
if st.sidebar.button("🔄 Sync knowledge base (LPNU + TG + rebuild index)", disabled=not st.session_state.online_mode):
    from pipelines.sync_all import sync_all

    channels = _parse_tg_channels(st.session_state.tg_channels)
    api_id = _safe_int(st.session_state.tg_api_id.strip()) if st.session_state.tg_api_id.strip() else None
    api_hash = st.session_state.tg_api_hash.strip() if st.session_state.tg_api_hash.strip() else None

    with st.spinner("Синхронізація знань... (LPNU сторінки можуть зайняти 1–3 хвилини)"):
        report = sync_all(
            api_id=api_id,
            api_hash=api_hash,
            channels=channels if (api_id and api_hash and channels) else None,
        )

    st.session_state.last_sync_report = report
    st.sidebar.success("Sync завершено ✅")
    st.sidebar.json(report)
    _maybe_load_index()

st.sidebar.divider()

# -----------------------------
# Advanced wipe
# -----------------------------
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


# -----------------------------
# Safety / reset
# -----------------------------
if st.sidebar.button("🧹 Очистити результати пошуку (UI)"):
    st.session_state.last_results = []
    st.success("Результати очищено.")

if st.sidebar.button("🧨 Повний скидання (очистити UI + креденшали)"):
    st.session_state.last_results = []
    st.session_state.vns_login = ""
    st.session_state.vns_password = ""
    st.session_state.tg_api_id = ""
    st.session_state.tg_api_hash = ""
    st.session_state.tg_phone = ""
    st.session_state.tg_code = ""
    st.session_state.tg_2fa = ""
    st.session_state.llm_api_key = ""
    st.session_state.last_sync_report = None
    st.success("Сесія очищена (без видалення файлів).")


# -----------------------------
# Main UI
# -----------------------------
st.title("🎓 IKNI Assistant — RAG MVP (Streamlit)")
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
        "щоб завантажити сторінки ІКНІ та побудувати індекс."
    )

if Path(CONFIG.faiss_index_path).exists() and not st.session_state.index_ready:
    _maybe_load_index()

if st.session_state.last_sync_report:
    with st.expander("📄 Останній Sync report", expanded=False):
        st.json(st.session_state.last_sync_report)

# Quick presets
st.subheader("⚡ Швидкі запити")
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
if qcol1.button("Хто директор ІКНІ?"):
    st.session_state.quick_query = "Хто директор ІКНІ?"
if qcol2.button("Коли створено ІКНІ?"):
    st.session_state.quick_query = "Коли створено ІКНІ?"
if qcol3.button("Керівництво ІКНІ"):
    st.session_state.quick_query = "Хто входить в керівництво ІКНІ?"
if qcol4.button("Що нового в pbikni?"):
    st.session_state.quick_query = "Що нового в Telegram каналі pbikni?"

default_query = st.session_state.get("quick_query", "")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("🔎 Запит")

    query = st.text_input("Введи питання", value=default_query)
    top_k = st.slider("Top-K джерел", min_value=1, max_value=10, value=CONFIG.top_k)

    use_llm = bool(st.session_state.online_mode and st.session_state.llm_enabled)
    if use_llm:
        st.caption(f"🤖 Генерація: **LLM ON** • provider=`{st.session_state.llm_provider}` • model=`{st.session_state.llm_model}`")
    else:
        st.caption("📌 Генерація: **LLM OFF** (offline summarizer)")

    ask_btn = st.button("Отримати відповідь", type="primary", use_container_width=True)

    if ask_btn:
        st.session_state.quick_query = ""  # reset
        if not query.strip():
            st.warning("Введи запит.")
        elif not st.session_state.index_ready:
            st.warning("Спершу завантаж або побудуй локальний FAISS індекс у сайдбарі.")
        else:
            from core.index import search_index

            index, meta = load_faiss_index(CONFIG.faiss_index_path, CONFIG.faiss_meta_path)

            results = search_index(
                query=query,
                index=index,
                chunks=meta,
                embed_model_name=CONFIG.embed_model_name,
                top_k=top_k,
            )
            st.session_state.last_results = results

            if use_llm:
                llm = _build_llm_settings()
                answer = make_answer_with_llm(query, results, llm)
            else:
                answer = make_answer_no_llm(query, results)

            st.markdown(answer)

with col2:
    st.subheader("📚 Джерела")
    if not st.session_state.last_results:
        st.info("Після запиту тут зʼявляться знайдені фрагменти (top-K).")
    else:
        rows = []
        for rank, (chunk, score) in enumerate(st.session_state.last_results, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Score": round(score, 4),
                    "Title": chunk.title,
                    "Type": chunk.source_type,
                    "Date": chunk.date,
                    "URL": chunk.url,
                    "Text (snippet)": (chunk.text[:180] + "…") if len(chunk.text) > 180 else chunk.text,
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

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
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
)

if Path(CONFIG.local_cache_path).exists():
    try:
        total_lines = sum(1 for _ in open(CONFIG.local_cache_path, "r", encoding="utf-8"))
        st.caption(f"📦 local_cache.jsonl: **{total_lines}** chunks (рядків)")
    except Exception:
        pass

st.caption(
    "Порада: Online mode → Telegram Auth → Sync → тестові запити. "
    "LLM робить відповідь структурованою, але офлайн теж працює."
)
