# 🎓 RAG Assistant — Multi-source RAG MVP (Streamlit + FAISS + LPNU + Telegram)

**RAG Assistant** — мінімально працююча версія (MVP) Retrieval-Augmented Generation системи для інформаційного супроводу студентів **Львівської політехніки**.  
Система збирає дані з відкритих джерел (**LPNU сторінки / Wiki / Telegram канали**), формує локальний архів знань (`local_cache.jsonl`), будує векторний індекс **FAISS** та відповідає на запити користувача через **семантичний retrieval** (offline) або **RAG + LLM** (online).

---

## ✅ Основні можливості (MVP)
- **🌐 Streamlit UI**: запити природною мовою + таблиця джерел (top-K)
- **⚡ ONLINE / OFFLINE режим**:
  - **ONLINE**: доступні синхронізація даних (**LPNU + Telegram + Wiki**) через crawler, LLM API
  - **OFFLINE**: працює тільки локальний архів та FAISS індекс
- **📥 Ingestion pipeline**:
  - **LPNU / Wiki сторінки** → асинхронний **crawler** → очищення тексту (trafilatura) → chunking → `local_cache.jsonl`
  - **Telegram канали** → last N messages → chunking → `local_cache.jsonl`
  - Підтримка будь-яких Telegram каналів (не тільки pbikni)
- **🔎 Vector Search**:
  - SentenceTransformers embeddings
  - FAISS IndexFlatIP (cosine similarity через normalize_embeddings)
- **🤖 RAG Answering**:
  - Offline summarizer (без LLM)
  - Optional LLM (OpenAI-compatible API: OpenAI / Groq / OpenRouter / Ollama / Custom)
- **🛡️ Безпека**:
  - VNS login/password та API ключі зберігаються тільки в `session_state` (оперативна пам’ять)
  - Можливість очищення вручну

## 🧱 Стек технологій

- Python 3.10+ (рекомендовано 3.10 або 3.11)
- Streamlit
- FAISS
- SentenceTransformers
- Telethon (Telegram ingestion)
- requests + trafilatura (LPNU ingestion)
- (optional) OpenAI-compatible LLM API

---

## 📂 Структура проєкту
C:.
|   .gitignore
|   app.py
|   eval_set.jsonl
|   README.md
|   requirements.txt
|   
+---assets
|       logo.png
|       
+---core
|       config.py
|       feedback.py
|       index.py
|       ingest_utils.py
|       llm.py
|       rag.py
|       rerank.py
|       security.py
|       sources.py
|       
+---data
|       index.faiss
|       index_meta.jsonl
|       local_cache.jsonl
|       
+---pipelines
|       crawl_lpnu.py
|       ingest_lpnu.py
|       ingest_site_resources.py
|       ingest_telegram.py
|       ingest_vns.py
|       sync_all.py
|       upload_ingest.py
|       
+---report
|       eval_results.csv
|       metrics.json
|       
+---scripts
        run_eval_and_plot.py

---

## 🚀 Швидкий старт

### 1) Клонування репозиторію
```bash
git clone <YOUR_REPO_URL>
cd RagSystem
```
2) Створення віртуального середовища

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\activate


Linux / macOS:

python -m venv .venv
source .venv/bin/activate

3) Встановлення залежностей
pip install -r requirements.txt


⚠️ Якщо виникають конфлікти numpy/faiss — рекомендовано:

Python 3.10 або 3.11

numpy<2.0

faiss-cpu==1.8.0.post1

▶️ Запуск застосунку
streamlit run app.py


Відкрий у браузері: http://localhost:8501

🔄 Перше наповнення бази (Sync)

Увімкни Online mode (ліва панель).

(Опційно) введи Telegram API ID + HASH для завантаження з телеграму

Натисни:

Sync knowledge base (LPNU + TG + rebuild index)
Це:

підтягне LPNU / Wiki сторінки
підтягне Telegram канали (якщо задано)
збереже чанки в local_cache.jsonl
перебудує FAISS індекс

Після цього можна задавати запити.

    Приклади запитів

Хто директор інституту?
Коли створено інститут?
Хто входить в керівництво?
Яка історія інституту?
Що нового в Telegram каналах?
Які партнери інституту?

    Підключення LLM (опційно)

У лівій панелі:

Увімкни Online mode

Відкрий LLM інтеграція

    Вибери провайдера:

openai

groq

openrouter

ollama

custom

Вибери модель зі списку (або вручну)

Введи API key (для ollama ключ не потрібен)

Натисни Test LLM

ℹ️ Temperature:

0.0–0.3 → коротко та стабільно

0.4–0.7 → баланс

0.8–1.0 → більш “креативно” та розгорнуто

    Telegram ingestion (pbikni)

Система підтримує завантаження повідомлень з Telegram каналу через Telethon.

Як отримати api_id та api_hash

Перейди на https://my.telegram.org/apps

Створи application (API Development Tools)

Скопіюй api_id та api_hash

⚠️ При першому запуску Telethon попросить код підтвердження (SMS/Telegram).
Сесія зберігається локально у data/tg_session.*

🔐 VNS інтеграція (підготовчий модуль)

У MVP реалізовано:

безпечне введення логіну/паролю VNS у UI

збереження тільки в оперативній пам’яті (session_state)

можливість очищення даних кнопкою

Повноцінний парсинг VNS та завантаження матеріалів планується як наступний етап.

🧠 Offline режим

У OFFLINE режимі система:

не виконує мережевих запитів

не синхронізує дані

не викликає LLM

працює тільки з локальним local_cache.jsonl та FAISS індексом

🧹 Очищення локальних даних

У UI є режим:

Wipe local cache + index
який видаляє:

local_cache.jsonl

FAISS індекс та метадані

Це корисно для тестів “з нуля”.

🛡️ Безпека

API ключі LLM та Telegram креденшали зберігаються тільки у session_state (пам’ять процесу Streamlit).

VNS логін/пароль не записуються у файли, не логуються, можуть бути очищені вручну.

data/, local_cache.jsonl, FAISS індекс та *.session не повинні пушитись у Git (див. .gitignore).

📌 Автор

Комарницький Богдан (ШІ-42)
Проєкт: створення модуля аналізу даних у інформаційній системі (RAG MVP)
Львівська політехніка