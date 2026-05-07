from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import faiss

from .index import filter_results, search_index
from .llm import LLMSettings
from .rag import (
    NO_KB_PHRASE,
    RAGAnswer,
    make_answer_no_llm_struct,
    make_answer_with_llm_struct,
    make_direct_answer_struct,
)
from .rerank import rerank_results
from .sources import SourceChunk

# --- Константи та Словники ---

VERIFIED_KB: Dict[str, Dict[str, str]] = {
    "university_rector": {
        "name": "Шаховська Наталія Богданівна",
        "role": "ректор",
        "entity_scope": "university",
    }
}

ABBREVIATIONS: Dict[str, str] = {
    "іард": "інститут архітектури та дизайну",
    "iard": "інститут архітектури та дизайну",
    "ікні": "інститут комп'ютерних наук та інформаційних технологій",
    "ikni": "інститут комп'ютерних наук та інформаційних технологій",
    "ігсн": "інститут гуманітарних та соціальних наук",
    "ihsn": "інститут гуманітарних та соціальних наук",
    "ігдг": "інститут геодезії",
    "igdg": "інститут геодезії",
    "іаду": "інститут адміністрування, державного управління та професійного розвитку",
    "iadu": "інститут адміністрування, державного управління та професійного розвитку",
    "ікта": "інститут комп'ютерних технологій, автоматики та метрології",
    "ikta": "інститут комп'ютерних технологій, автоматики та метрології",
}

INSTITUTE_CANONICAL: Dict[str, str] = {
    "iard": "інститут архітектури та дизайну",
    "ikni": "інститут комп'ютерних наук та інформаційних технологій",
    "ikta": "інститут комп'ютерних технологій, автоматики та метрології",
    "ihsn": "інститут гуманітарних та соціальних наук",
    "igdg": "інститут геодезії",
    "iadu": "інститут адміністрування, державного управління та професійного розвитку",
    "inem": "інститут економіки і менеджменту",
    "ibib": "інститут будівництва та інженерних систем",
}

INSTITUTE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "iard": ("іард", "iard", "архітектури та дизайну"),
    "ikni": ("ікні", "ikni", "комп'ютерних наук та інформаційних технологій"),
    "ikta": ("ікта", "ikta", "комп'ютерних технологій, автоматики та метрології"),
    "ihsn": ("ігсн", "ihsn", "гуманітарних та соціальних наук"),
    "igdg": ("ігдг", "igdg", "ihdh", "геодез"),
    "iadu": ("іаду", "iadu", "державного управління"),
    "inem": ("інем", "inem", "економіки і менеджменту"),
    "ibib": ("ібіб", "ibib", "будівництва та інженерних систем"),
}

# --- Хінти та Регулярні вирази ---

PERSON_HINTS = ("хто", "директор", "ректор", "керівник", "очолює", "декан", "завідувач", "керує")
LOCATION_HINTS = ("де", "адреса", "корпус", "кабінет", "аудитор", "знаход", "розташ")
CONTACT_HINTS = ("контакт", "телефон", "email", "e-mail", "пошта", "зв'яз")
VNS_HINTS = ("внс", "розклад", "оцінки", "модуль", "силабус", "курс", "кабінет студента")
LOCAL_HINTS = ("pdf", "методич", "презентац", "лаборатор", "лекц", "документ")
NEWS_HINTS = ("новини", "події", "оголошення", "анонс")

PERSON_BOOST_TERMS = ("директор", "ректор", "керівництво", "очолює", "завідувач", "декан")
LOCATION_BOOST_TERMS = ("адреса", "вул.", "вулиця", "корпус", "кабінет", "аудитор", "розташований", "знаходиться")
CONTACT_BOOST_TERMS = ("телефон", "email", "e-mail", "пошта", "контакти")

SOURCE_TRUST_WEIGHTS: Dict[str, float] = {
    "local": 0.98,
    "vns": 0.95,
    "lpnu": 0.92,
    "resource": 0.86,
    "lpnu_resource": 0.86,
    "tg": 0.68,
}

NAME_RE = r"[А-ЯІЇЄҐ][а-яіїєґ'’\-]+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ'’\-]+){1,2}"
ADDRESS_RE = r"(?:вул\.?|вулиця|м\.)\s*[А-ЯІЇЄҐA-Z0-9][^,\n]{2,90}(?:,\s*\d+[А-Яа-яA-Za-z/]*)?"


# --- Моделі Даних ---

@dataclass
class QueryAnalysis:
    original_query: str
    expanded_query: str
    question_type: str
    adaptive_top_k: int
    abbreviations: Dict[str, str] = field(default_factory=dict)
    required_entities: List[str] = field(default_factory=dict)
    institute_code: Optional[str] = None
    institute_name: Optional[str] = None
    intent: str = "generic"
    entity_scope: str = "generic"
    keyword_boost_terms: List[str] = field(default_factory=list)
    preferred_sources: Dict[str, float] = field(default_factory=dict)
    context_limit: int = 3


@dataclass
class ExtractionResult:
    found: bool
    answer_text: str = ""
    confidence: str = "Low"
    explanation: Optional[str] = None
    used_sources: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RetrievalBundle:
    analysis: QueryAnalysis
    candidates: List[Tuple[SourceChunk, float]]
    results: List[Tuple[SourceChunk, float]]
    context_results: List[Tuple[SourceChunk, float]]
    extraction: Optional[ExtractionResult] = None
    entity_candidates: List[Dict[str, Any]] = field(default_factory=list)
    accepted_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    final_decision: str = ""
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerQueryConfig:
    embed_model_name: str
    min_score: float = 0.2
    keyword_filter: bool = False
    allowed_types: Optional[Tuple[str, ...]] = None
    allowed_doc_ids: Optional[Tuple[str, ...]] = None
    use_reranker: bool = False
    reranker_model: Optional[str] = None
    reranker_top_n: int = 20
    use_query_expansion: bool = True
    use_adaptive_top_k: bool = True
    use_hybrid: bool = True
    top_k_override: Optional[int] = None
    use_post_boosts: bool = True
    use_extraction: bool = True
    use_institute_filter: bool = True
    use_rules: bool = True
    llm: Optional[LLMSettings] = None


# --- Допоміжні функції ---

VALID_DIRECTOR_ROLE_PHRASE = "директор інституту"
REJECT_ROLE_TERMS = ("заступник", "адміністратор", "комісії", "технічний", "гуртожиток")
NOISE_TERMS = ("гуртожит", "студмістечко", "адміністратор", "комісії", "технічн", "підтримк")
LEADERSHIP_TERMS = ("керівництво", "дирекція", "директор", "керівник", "очолює", "інститут")


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-zа-яіїєґ0-9/\-_' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _query_tokens(query: str) -> List[str]:
    return re.findall(r"[a-zа-яіїєґ0-9']+", _normalize_text(query))


def _detect_institute_from_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = _normalize_text(query)
    for code, aliases in INSTITUTE_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return code, INSTITUTE_CANONICAL.get(code)
    return None, None


def _detect_chunk_institute(chunk: SourceChunk) -> Tuple[Optional[str], Optional[str]]:
    extra = chunk.extra or {}
    code = (extra.get("institute_code") or "").strip().lower()
    name = (extra.get("institute_name") or "").strip().lower()

    if code in INSTITUTE_CANONICAL:
        return code, INSTITUTE_CANONICAL[code]

    if name:
        for cand_code, canonical in INSTITUTE_CANONICAL.items():
            if _normalize_text(canonical) in _normalize_text(name):
                return cand_code, canonical

    haystack = _normalize_text(f"{chunk.url or ''} {chunk.title or ''} {chunk.text or ''}")
    for cand_code, aliases in INSTITUTE_ALIASES.items():
        if any(alias in haystack for alias in aliases):
            return cand_code, INSTITUTE_CANONICAL.get(cand_code)
    return None, None


def _attach_institute_metadata(chunk: SourceChunk) -> SourceChunk:
    extra = chunk.extra or {}
    code, name = _detect_chunk_institute(chunk)
    if code and not extra.get("institute_code"):
        extra["institute_code"] = code
    if name and not extra.get("institute_name"):
        extra["institute_name"] = name
    chunk.extra = extra
    return chunk


def source_trust_score(chunk: SourceChunk) -> float:
    extra = chunk.extra or {}
    explicit = extra.get("source_trust")
    if explicit is not None:
        try:
            return float(explicit)
        except Exception:
            pass
    return SOURCE_TRUST_WEIGHTS.get((chunk.source_type or "").lower(), 0.72)


# --- Основна Логіка Аналізу та Пошуку ---

def analyze_query(query: str) -> QueryAnalysis:
    normalized = _normalize_text(query)
    tokens = _query_tokens(query)
    expansions: Dict[str, str] = {}
    required_entities: List[str] = []

    institute_code, institute_name = _detect_institute_from_query(query)

    for token in tokens:
        if token in ABBREVIATIONS:
            expansions[token] = ABBREVIATIONS[token]
            required_entities.append(ABBREVIATIONS[token])

    if institute_name:
        required_entities.append(institute_name)

    intent = "generic"
    entity_scope = "generic"

    if "хто" in normalized and "ректор" in normalized and "директор" not in normalized:
        intent = "university_rector"
        entity_scope = "university"
    elif "хто" in normalized and "директор" in normalized:
        intent = "institute_director"
        entity_scope = "institute"

    if any(hint in normalized for hint in PERSON_HINTS):
        question_type = "person"
        adaptive_top_k = 2
        boost_terms = list(PERSON_BOOST_TERMS)
    elif any(hint in normalized for hint in LOCATION_HINTS):
        question_type = "location"
        adaptive_top_k = 3
        boost_terms = list(LOCATION_BOOST_TERMS)
    elif any(hint in normalized for hint in CONTACT_HINTS):
        question_type = "contact"
        adaptive_top_k = 3
        boost_terms = list(CONTACT_BOOST_TERMS)
    else:
        question_type = "general"
        adaptive_top_k = 5
        boost_terms = []

    preferred_sources: Dict[str, float] = {"lpnu": 0.05}
    if any(hint in normalized for hint in VNS_HINTS):
        preferred_sources["vns"] = 0.22
    if any(hint in normalized for hint in LOCAL_HINTS):
        preferred_sources["local"] = 0.18
    if any(hint in normalized for hint in NEWS_HINTS):
        preferred_sources["tg"] = 0.15
        preferred_sources["lpnu"] = 0.12

    expanded_parts = [query.strip()]
    expanded_parts.extend(expansions.values())
    if institute_name:
        expanded_parts.append(institute_name)

    if question_type == "person":
        expanded_parts.append("директор керівництво ректор декан")
    elif question_type == "location":
        expanded_parts.append("адреса корпус кабінет аудиторія")
    elif question_type == "contact":
        expanded_parts.append("контакти телефон email e-mail")

    expanded_query = " ".join(part for part in expanded_parts if part).strip()

    return QueryAnalysis(
        original_query=query,
        expanded_query=expanded_query or query,
        question_type=question_type,
        adaptive_top_k=adaptive_top_k,
        abbreviations=expansions,
        required_entities=list(dict.fromkeys(required_entities)),
        institute_code=institute_code,
        institute_name=institute_name,
        intent=intent,
        entity_scope=entity_scope,
        keyword_boost_terms=boost_terms,
        preferred_sources=preferred_sources,
        context_limit=2 if question_type == "person" else 3,
    )


def analyze_query_with_options(
        query: str,
        *,
        use_query_expansion: bool = True,
        use_adaptive_top_k: bool = True
) -> QueryAnalysis:
    analysis = analyze_query(query)
    if not use_query_expansion:
        analysis.expanded_query = analysis.original_query
        analysis.abbreviations = {}
        analysis.required_entities = []
    if not use_adaptive_top_k:
        analysis.adaptive_top_k = 5
        analysis.context_limit = 5
    return analysis


def _keyword_overlap(analysis: QueryAnalysis, chunk: SourceChunk) -> int:
    keywords = {token for token in _query_tokens(analysis.expanded_query) if len(token) >= 4}
    if not keywords:
        return 0
    haystack = _normalize_text(f"{chunk.title or ''} {chunk.url or ''} {chunk.text or ''}")
    return sum(1 for keyword in keywords if keyword in haystack)


def _entity_hits(analysis: QueryAnalysis, chunk: SourceChunk) -> int:
    if not analysis.required_entities:
        return 0
    haystack = _normalize_text(f"{chunk.title or ''} {chunk.url or ''} {chunk.text or ''}")
    return sum(1 for entity in analysis.required_entities if _normalize_text(entity) in haystack)


def _post_retrieval_score(analysis: QueryAnalysis, chunk: SourceChunk, score: float) -> float:
    boosted = float(score)
    url = (chunk.url or "").lower()
    haystack = _normalize_text(f"{chunk.title or ''} {chunk.url or ''} {chunk.text or ''}")

    boosted += min(_keyword_overlap(analysis, chunk) * 0.04, 0.16)
    boosted += min(_entity_hits(analysis, chunk) * 0.18, 0.36)

    for term in analysis.keyword_boost_terms:
        if term in haystack:
            boosted += 0.08

    chunk = _attach_institute_metadata(chunk)
    chunk_institute = ((chunk.extra or {}).get("institute_code") or "").strip().lower()

    if analysis.institute_code and chunk_institute:
        boosted += 0.45 if chunk_institute == analysis.institute_code else -0.70

    if any(k in url for k in ["kerivnytstvo", "dyrektsiia", "administration"]):
        boosted += 0.5

    preferred_bonus = analysis.preferred_sources.get((chunk.source_type or "").lower(), 0.0)
    boosted += preferred_bonus
    boosted += max(source_trust_score(chunk) - 0.7, 0.0) * 0.1
    return boosted


def _dedupe_by_url(results: Sequence[Tuple[SourceChunk, float]]) -> List[Tuple[SourceChunk, float]]:
    unique: List[Tuple[SourceChunk, float]] = []
    seen = set()
    for chunk, score in results:
        extra = chunk.extra or {}
        key = ((chunk.url or "").rstrip("/"), extra.get("doc_id"), chunk.title or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append((chunk, score))
    return unique


def _filter_by_institute(
        analysis: QueryAnalysis,
        results: Sequence[Tuple[SourceChunk, float]],
) -> List[Tuple[SourceChunk, float]]:
    if not analysis.institute_code:
        return list(results)
    filtered: List[Tuple[SourceChunk, float]] = []
    for chunk, score in results:
        chunk = _attach_institute_metadata(chunk)
        code = ((chunk.extra or {}).get("institute_code") or "").strip().lower()
        if code == analysis.institute_code:
            filtered.append((chunk, score))
    return filtered


def _apply_entity_scope_constraints(
        analysis: QueryAnalysis,
        results: Sequence[Tuple[SourceChunk, float]],
) -> List[Tuple[SourceChunk, float]]:
    constrained: List[Tuple[SourceChunk, float]] = []
    for chunk, score in results:
        text = _normalize_text(f"{chunk.title or ''} {chunk.url or ''} {chunk.text or ''}")

        if analysis.entity_scope == "university":
            if "інститут" in text and "львівська політехніка" not in text and "університет" not in text:
                continue
            constrained.append((chunk, score))
            continue

        if analysis.entity_scope == "institute":
            if analysis.institute_code:
                code = ((chunk.extra or {}).get("institute_code") or "").strip().lower()
                if code and code != analysis.institute_code:
                    continue
            if "ректор" in text and "директор" not in text:
                continue
            constrained.append((chunk, score))
            continue

        constrained.append((chunk, score))
    return constrained


def _is_noisy_chunk(chunk: SourceChunk) -> bool:
    haystack = _normalize_text(f"{chunk.title or ''} {chunk.url or ''} {chunk.text or ''}")
    if any(term in haystack for term in NOISE_TERMS):
        return True
    if not any(term in haystack for term in LEADERSHIP_TERMS):
        return True
    return False


def _are_similar_chunks(left: SourceChunk, right: SourceChunk) -> bool:
    lt = set(_query_tokens((left.text or "")[:500]))
    rt = set(_query_tokens((right.text or "")[:500]))
    if not lt or not rt:
        return False
    overlap = len(lt & rt) / max(len(lt | rt), 1)
    return overlap >= 0.75


def clean_context_results(bundle: RetrievalBundle) -> List[Tuple[SourceChunk, float]]:
    analysis = bundle.analysis
    institute_scoped = _filter_by_institute(analysis, bundle.results)
    candidates = institute_scoped if institute_scoped else bundle.results

    selected: List[Tuple[SourceChunk, float]] = []
    context_institute: Optional[str] = analysis.institute_code if institute_scoped else None

    for chunk, score in candidates:
        chunk = _attach_institute_metadata(chunk)
        if analysis.intent == "institute_director" and _is_noisy_chunk(chunk):
            continue

        code = ((chunk.extra or {}).get("institute_code") or "").strip().lower()
        if context_institute and code and code != context_institute:
            continue
        if not context_institute and code and analysis.institute_code and code == analysis.institute_code:
            context_institute = code
        elif context_institute and not code:
            continue

        overlap = _keyword_overlap(analysis, chunk)
        entity_hits = _entity_hits(analysis, chunk)

        if analysis.required_entities and entity_hits == 0:
            if overlap == 0:
                continue
        elif overlap == 0 and analysis.question_type != "general":
            continue

        if any(_are_similar_chunks(chunk, prev_chunk) for prev_chunk, _ in selected):
            continue

        selected.append((chunk, score))
        if len(selected) >= analysis.context_limit:
            break

    fallback = candidates[: analysis.context_limit]
    return selected or fallback


def retrieve_for_query(
        *,
        query: str,
        index: faiss.Index,
        chunks: List[SourceChunk],
        embed_model_name: str,
        min_score: float = 0.2,
        keyword_filter: bool = False,
        allowed_types: Optional[Set[str]] = None,
        allowed_doc_ids: Optional[Set[str]] = None,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        reranker_top_n: int = 20,
        use_query_expansion: bool = True,
        use_adaptive_top_k: bool = True,
        use_hybrid: bool = True,
        top_k_override: Optional[int] = None,
        use_post_boosts: bool = True,
        use_extraction: bool = True,
        use_institute_filter: bool = True,
) -> RetrievalBundle:
    timings: Dict[str, float] = {}
    started_at = time.perf_counter()
    analysis = analyze_query_with_options(
        query,
        use_query_expansion=use_query_expansion,
        use_adaptive_top_k=use_adaptive_top_k,
    )

    requested_top_k = top_k_override or analysis.adaptive_top_k
    retrieval_started = time.perf_counter()
    candidates = search_index(
        query=analysis.expanded_query,
        index=index,
        chunks=chunks,
        embed_model_name=embed_model_name,
        top_k=max(requested_top_k, 5),
        min_score=min_score,
        keyword_filter=keyword_filter or bool(analysis.required_entities),
        internal_k=max(requested_top_k * 20, 80),
        use_hybrid=use_hybrid,
        use_query_boosts=use_post_boosts,
    )
    timings["search_ms"] = round((time.perf_counter() - retrieval_started) * 1000.0, 3)

    if allowed_types:
        candidates = filter_results(candidates, allowed_types=allowed_types)

    if not candidates:
        fallback_started = time.perf_counter()
        candidates = search_index(
            query=analysis.expanded_query,
            index=index,
            chunks=chunks,
            embed_model_name=embed_model_name,
            top_k=max(requested_top_k, 5),
            min_score=min_score,
            keyword_filter=False,
            internal_k=max(requested_top_k * 20, 80),
            use_hybrid=use_hybrid,
            use_query_boosts=use_post_boosts,
        )
        timings["fallback_search_ms"] = round((time.perf_counter() - fallback_started) * 1000.0, 3)

    if allowed_doc_ids:
        candidates = filter_results(candidates, allowed_doc_ids=allowed_doc_ids)

    candidates = [(_attach_institute_metadata(chunk), score) for chunk, score in candidates]

    if use_institute_filter and analysis.institute_code:
        institute_filtered = _filter_by_institute(analysis, candidates)
        if institute_filtered:
            candidates = institute_filtered

    if use_post_boosts:
        rescored = [(chunk, _post_retrieval_score(analysis, chunk, score)) for chunk, score in candidates]
    else:
        rescored = list(candidates)

    rescored.sort(key=lambda item: item[1], reverse=True)
    rescored = _dedupe_by_url(rescored)

    if use_reranker and reranker_model:
        rerank_started = time.perf_counter()
        rerank_pool = rescored[: max(reranker_top_n, requested_top_k * 4)]
        results = rerank_results(
            query=analysis.expanded_query,
            results=rerank_pool,
            model_name=reranker_model,
            top_k=requested_top_k,
        )
        timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000.0, 3)
    else:
        results = rescored[: requested_top_k]
        timings["rerank_ms"] = 0.0

    results = _dedupe_by_url(results)

    bundle = RetrievalBundle(
        analysis=analysis,
        candidates=rescored,
        results=results,
        context_results=[],
        extraction=None,
        entity_candidates=[],
        timings=timings,
    )
    bundle.context_results = clean_context_results(bundle)

    if use_extraction:
        extraction_started = time.perf_counter()
        bundle.extraction = extract_answer_before_llm(query, bundle.context_results, analysis)
        bundle.timings["extraction_ms"] = round((time.perf_counter() - extraction_started) * 1000.0, 3)
    else:
        bundle.timings["extraction_ms"] = 0.0

    if analysis.question_type == "person":
        bundle.entity_candidates = _extract_person_candidates(query, bundle.context_results, analysis)

    bundle.timings["total_retrieval_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
    return bundle


# --- Витягування сутностей (Person/Location/Contact) ---

def _person_answer_suffix(query: str, analysis: QueryAnalysis) -> str:
    if analysis.institute_code:
        return (analysis.institute_code or "").upper()
    if analysis.institute_name:
        return analysis.institute_name
    upper_query = query.strip()
    for abbr in analysis.abbreviations:
        if abbr.upper() in upper_query.upper():
            return abbr.upper()
    if analysis.required_entities:
        return analysis.required_entities[0]
    return "інституту"


def _person_role_label(query: str) -> str:
    normalized = _normalize_text(query)
    if "ректор" in normalized:
        return "ректор"
    if "декан" in normalized:
        return "декан"
    if "завідувач" in normalized:
        return "завідувач"
    if "керів" in normalized or "очолює" in normalized or "керує" in normalized:
        return "керівник"
    return "директор"


def _requested_person_role(query: str) -> str:
    normalized = _normalize_text(query)
    if "ректор" in normalized:
        return "rector"
    if "заступник" in normalized:
        return "deputy"
    if "декан" in normalized:
        return "dean"
    if "завідувач" in normalized:
        return "head"
    if "директор" in normalized:
        return "director"
    return "leader"


def _role_from_text(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    if any(x in normalized for x in ["заступник директора", "заступниця директора", "заступник"]):
        return "deputy"
    if "директор" in normalized:
        return "director"
    if "ректор" in normalized:
        return "rector"
    if "декан" in normalized:
        return "dean"
    if "завідувач" in normalized:
        return "head"
    if any(x in normalized for x in ["керівник", "очолює", "керує"]):
        return "leader"
    return None


def _role_label_ua(role_code: str) -> str:
    labels = {
        "director": "директор",
        "deputy": "заступник директора",
        "rector": "ректор",
        "dean": "декан",
        "head": "завідувач",
        "leader": "керівник",
    }
    return labels.get(role_code, "керівник")


def _role_match_score(requested_role: str, candidate_role: Optional[str]) -> float:
    if not candidate_role:
        return -0.05
    if requested_role == "leader":
        return 0.12 if candidate_role in {"director", "rector", "dean", "head", "leader"} else -0.06
    if requested_role == candidate_role:
        return 0.6
    if requested_role == "director" and candidate_role == "deputy":
        return -0.75
    if requested_role in {"rector", "dean", "head"} and candidate_role == "deputy":
        return -0.45
    return -0.25


def _query_relevance_score(analysis: QueryAnalysis, chunk: SourceChunk) -> float:
    overlap = _keyword_overlap(analysis, chunk)
    entity_hits = _entity_hits(analysis, chunk)
    return min(overlap * 0.03, 0.15) + min(entity_hits * 0.18, 0.36)


def _extract_person_candidates(
        query: str,
        results: Sequence[Tuple[SourceChunk, float]],
        analysis: QueryAnalysis,
) -> List[Dict[str, Any]]:
    requested_role = _requested_person_role(query)
    raw_candidates: List[Dict[str, Any]] = []

    for idx, (chunk, retrieval_score) in enumerate(results, start=1):
        chunk = _attach_institute_metadata(chunk)
        chunk_institute_code = ((chunk.extra or {}).get("institute_code") or "").strip().lower()
        chunk_institute_name = ((chunk.extra or {}).get("institute_name") or "").strip()

        lines = [line.strip() for line in re.split(r"[\n\r]+", f"{chunk.title or ''}\n{chunk.text or ''}") if
                 line.strip()]

        for line_idx, line in enumerate(lines):
            names = list(dict.fromkeys(re.findall(NAME_RE, line)))
            if not names:
                continue

            window = " ".join(lines[max(0, line_idx - 1): min(len(lines), line_idx + 2)])
            role_text = f"{line} {window}"
            role_code = _role_from_text(role_text)

            if not role_code:
                continue

            for name in names:
                if len(name.split()) < 2:
                    continue

                role_score = _role_match_score(requested_role, role_code)
                if analysis.intent == "institute_director" and role_code != "director":
                    continue
                if analysis.intent == "university_rector" and role_code != "rector":
                    continue

                relevance_score = _query_relevance_score(analysis, chunk)
                institute_score = 0.0
                if analysis.institute_code and chunk_institute_code:
                    institute_score = 0.55 if chunk_institute_code == analysis.institute_code else -0.9

                total_score = float(retrieval_score) + role_score + relevance_score + institute_score

                raw_candidates.append({
                    "name": re.sub(r"\s+", " ", name).strip(" ,.-"),
                    "role_code": role_code,
                    "role_label": _role_label_ua(role_code),
                    "institute_code": chunk_institute_code or None,
                    "institute_name": chunk_institute_name or None,
                    "source_rank": idx,
                    "retrieval_score": round(float(retrieval_score), 4),
                    "role_score": round(role_score, 4),
                    "query_relevance_score": round(relevance_score, 4),
                    "institute_score": round(institute_score, 4),
                    "total_score": round(total_score, 4),
                    "context_line": line[:220],
                    "context_window": role_text[:320],
                })

    best_by_name: Dict[str, Dict[str, Any]] = {}
    for cand in raw_candidates:
        key = _normalize_text(cand["name"])
        if key not in best_by_name or cand["total_score"] > best_by_name[key]["total_score"]:
            best_by_name[key] = cand

    candidates = sorted(best_by_name.values(), key=lambda item: item["total_score"], reverse=True)

    if analysis.institute_code:
        institute_only = [cand for cand in candidates if cand.get("institute_code") == analysis.institute_code]
        if institute_only:
            candidates = institute_only

    return candidates[:20]


def _validate_director_candidates(
        analysis: QueryAnalysis,
        candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for cand in candidates:
        line = _normalize_text(cand.get("context_line") or "")
        window = _normalize_text(cand.get("context_window") or line)
        reject_reason = None

        if cand.get("role_code") != "director":
            reject_reason = "role_not_director"
        if any(x in window for x in ["директор", "очолює", "дирекція", "керівництво"]):
            # валідно
            pass
        elif any(term in window for term in REJECT_ROLE_TERMS):
            reject_reason = "blacklisted_role_context"
        elif "директор" not in window and "очолює" not in window:
            reject_reason = "missing_director_context"
        elif analysis.institute_code and cand.get("institute_code") != analysis.institute_code:
            reject_reason = "wrong_institute"

        if reject_reason:
            rejected.append({**cand, "reject_reason": reject_reason})
        else:
            accepted.append(cand)

    return accepted, rejected


def _known_answer_for_intent(analysis: QueryAnalysis) -> Optional[str]:
    known = VERIFIED_KB.get(analysis.intent)
    if not known:
        return None
    if analysis.intent == "university_rector":
        return f"{known['name']} — {known['role']} Львівської політехніки."
    return None


def _extract_person_answer(
        query: str,
        results: Sequence[Tuple[SourceChunk, float]],
        analysis: QueryAnalysis
) -> ExtractionResult:
    candidates = _extract_person_candidates(query, results, analysis)

    if analysis.intent == "institute_director":
        accepted, rejected = _validate_director_candidates(analysis, candidates)
        if not accepted:
            return ExtractionResult(
                found=False,
                warnings=["director_validation_failed", f"rejected_candidates={len(rejected)}"],
            )
        top = sorted(accepted, key=lambda item: item.get("total_score", 0.0), reverse=True)[0]
        inst_name = analysis.institute_name or (top.get("institute_name") or "інституту")
        inst_name = inst_name[:1].upper() + inst_name[1:] if inst_name else "інституту"

        confidence = "High" if top["source_rank"] == 1 and top["total_score"] >= 0.5 else "Medium"
        answer_text = f"{top['name']} — директор {inst_name}."

        return ExtractionResult(
            found=True,
            answer_text=answer_text,
            confidence=confidence,
            explanation="Кандидат пройшов жорстку валідацію ролі 'директор інституту'.",
            used_sources=[int(top["source_rank"])],
            warnings=[],
        )

    if candidates:
        top = candidates[0]
        confidence = "High" if top["source_rank"] == 1 and top["total_score"] >= 0.5 else "Medium"
        answer_text = f"{top['name']} — {top['role_label']} {_person_answer_suffix(query, analysis)}."
        return ExtractionResult(
            found=True,
            answer_text=answer_text,
            confidence=confidence,
            explanation="Відповідь витягнута структурним парсером і обрана на рівні сутностей.",
            used_sources=[int(top["source_rank"])],
        )

    return ExtractionResult(found=False)


def _extract_location_answer(results: Sequence[Tuple[SourceChunk, float]], analysis: QueryAnalysis) -> ExtractionResult:
    address_re = re.compile(ADDRESS_RE, re.IGNORECASE)
    for idx, (chunk, _) in enumerate(results, start=1):
        haystack = f"{chunk.title or ''}\n{chunk.text or ''}"
        match = address_re.search(haystack)
        if not match:
            continue
        address = re.sub(r"\s+", " ", match.group(0)).strip(" ,.")
        confidence = "High" if idx == 1 else "Medium"
        return ExtractionResult(
            found=True,
            answer_text=f"Адреса: {address}.",
            confidence=confidence,
            explanation="Адресу витягнуто напряму з джерела.",
            used_sources=[idx],
        )
    return ExtractionResult(found=False)


def _extract_contact_answer(results: Sequence[Tuple[SourceChunk, float]], analysis: QueryAnalysis) -> ExtractionResult:
    phone_matcher = re.compile(r"(\+?\d[\d\-\(\)\s]{7,}\d)")
    email_matcher = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    for idx, (chunk, _) in enumerate(results, start=1):
        text = f"{chunk.title or ''}\n{chunk.text or ''}"
        phones = list(dict.fromkeys(phone_matcher.findall(text)))
        emails = list(dict.fromkeys(email_matcher.findall(text)))
        if not phones and not emails:
            continue
        parts = []
        if phones:
            parts.append(f"Телефон: {phones[0]}")
        if emails:
            parts.append(f"Email: {emails[0]}")
        return ExtractionResult(
            found=True,
            answer_text=". ".join(parts) + ".",
            confidence="Medium" if idx > 1 else "High",
            explanation="Контактні дані витягнуто напряму з джерела.",
            used_sources=[idx],
        )
    return ExtractionResult(found=False)


def extract_answer_before_llm(
        query: str,
        results: Sequence[Tuple[SourceChunk, float]],
        analysis: Optional[QueryAnalysis] = None,
) -> ExtractionResult:
    analysis = analysis or analyze_query(query)
    if not results:
        return ExtractionResult(found=False, warnings=["no_results"])

    if analysis.question_type == "person":
        extracted = _extract_person_answer(query, results, analysis)
        if extracted.found:
            return extracted
        known_answer = _known_answer_for_intent(analysis)
        if known_answer:
            return ExtractionResult(
                found=True,
                answer_text=known_answer,
                confidence="High",
                explanation="Відповідь взято з перевіреної бази знань.",
                used_sources=[],
                warnings=["verified_kb_fallback"],
            )
        return extracted

    if analysis.question_type == "location":
        return _extract_location_answer(results, analysis)
    if analysis.question_type == "contact":
        return _extract_contact_answer(results, analysis)

    return ExtractionResult(found=False)


# --- Основна функція відповіді ---

def answer_query_with_config(
        *,
        query: str,
        index: faiss.Index,
        chunks: List[SourceChunk],
        config: AnswerQueryConfig,
) -> Tuple[RAGAnswer, RetrievalBundle]:
    allowed_types = set(config.allowed_types) if config.allowed_types else None
    allowed_doc_ids = set(config.allowed_doc_ids) if config.allowed_doc_ids else None
    return answer_query(
        query=query,
        index=index,
        chunks=chunks,
        embed_model_name=config.embed_model_name,
        llm=config.llm,
        min_score=config.min_score,
        keyword_filter=config.keyword_filter,
        allowed_types=allowed_types,
        allowed_doc_ids=allowed_doc_ids,
        use_reranker=config.use_reranker,
        reranker_model=config.reranker_model,
        reranker_top_n=config.reranker_top_n,
        use_query_expansion=config.use_query_expansion,
        use_adaptive_top_k=config.use_adaptive_top_k,
        use_hybrid=config.use_hybrid,
        top_k_override=config.top_k_override,
        use_post_boosts=config.use_post_boosts,
        use_extraction=config.use_extraction,
        use_institute_filter=config.use_institute_filter,
        use_rules=config.use_rules,
    )


def answer_query(
        *,
        query: str,
        index: faiss.Index,
        chunks: List[SourceChunk],
        embed_model_name: str,
        llm: Optional[LLMSettings] = None,
        min_score: float = 0.2,
        keyword_filter: bool = False,
        allowed_types: Optional[Set[str]] = None,
        allowed_doc_ids: Optional[Set[str]] = None,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        reranker_top_n: int = 20,
        use_query_expansion: bool = True,
        use_adaptive_top_k: bool = True,
        use_hybrid: bool = True,
        top_k_override: Optional[int] = None,
        use_post_boosts: bool = True,
        use_extraction: bool = True,
        use_institute_filter: bool = True,
        use_rules: bool = True,
) -> Tuple[RAGAnswer, RetrievalBundle]:
    bundle = retrieve_for_query(
        query=query,
        index=index,
        chunks=chunks,
        embed_model_name=embed_model_name,
        min_score=min_score,
        keyword_filter=keyword_filter,
        allowed_types=allowed_types,
        allowed_doc_ids=allowed_doc_ids,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        reranker_top_n=reranker_top_n,
        use_query_expansion=use_query_expansion,
        use_adaptive_top_k=use_adaptive_top_k,
        use_hybrid=use_hybrid,
        top_k_override=top_k_override,
        use_post_boosts=use_post_boosts,
        use_extraction=use_extraction,
        use_institute_filter=use_institute_filter if use_rules else False,
    )

    if use_rules:
        bundle.results = _apply_entity_scope_constraints(bundle.analysis, bundle.results)
        bundle.context_results = _apply_entity_scope_constraints(bundle.analysis, bundle.context_results)

    if not bundle.context_results:
        bundle.context_results = bundle.results[: max(bundle.analysis.context_limit, 1)]

    known_priority = _known_answer_for_intent(bundle.analysis)
    if known_priority and bundle.analysis.intent == "university_rector":
        answer = make_direct_answer_struct(
            query=query,
            answer_text=known_priority,
            retrieved=bundle.context_results,
            used_sources=[],
            confidence="High",
            explanation="Пріоритетна відповідь з перевіреної бази знань.",
            warnings=["verified_kb_priority"],
            metrics={
                "question_type": bundle.analysis.question_type,
                "intent": bundle.analysis.intent,
                "entity_scope": bundle.analysis.entity_scope,
                "retrieved_count": len(bundle.context_results),
                "entity_candidates": len(bundle.entity_candidates),
            },
        )
        return answer, bundle

    if bundle.analysis.intent == "institute_director":
        accepted, rejected = _validate_director_candidates(bundle.analysis, bundle.entity_candidates)
        bundle.accepted_candidates = accepted
        bundle.rejected_candidates = rejected
        bundle.final_decision = "accepted_director_candidate" if accepted else "no_valid_director_candidate"

    if bundle.extraction and bundle.extraction.found:
        answer = make_direct_answer_struct(
            query=query,
            answer_text=bundle.extraction.answer_text,
            retrieved=bundle.context_results,
            used_sources=bundle.extraction.used_sources,
            confidence=bundle.extraction.confidence,
            explanation=bundle.extraction.explanation,
            warnings=bundle.extraction.warnings,
            metrics={
                "question_type": bundle.analysis.question_type,
                "intent": bundle.analysis.intent,
                "entity_scope": bundle.analysis.entity_scope,
                "adaptive_top_k": bundle.analysis.adaptive_top_k,
                "retrieved_count": len(bundle.context_results),
                "query_expanded": bundle.analysis.expanded_query != bundle.analysis.original_query,
                "entity_candidates": len(bundle.entity_candidates),
            },
        )
        return answer, bundle

    if bundle.analysis.intent == "institute_director" and use_rules:
        bundle.final_decision = bundle.final_decision or "rejected_no_valid_director"
        no_answer = make_direct_answer_struct(
            query=query,
            answer_text=NO_KB_PHRASE,
            retrieved=bundle.context_results,
            used_sources=[],
            confidence="Low",
            explanation="Немає кандидата з валідною роллю 'директор інституту'.",
            warnings=["strict_director_no_answer"],
            metrics={
                "question_type": bundle.analysis.question_type,
                "intent": bundle.analysis.intent,
                "entity_scope": bundle.analysis.entity_scope,
                "entity_candidates": len(bundle.entity_candidates),
                "accepted_candidates": len(bundle.accepted_candidates),
                "rejected_candidates": len(bundle.rejected_candidates),
            },
        )
        return no_answer, bundle

    if llm and llm.enabled:
        answer = make_answer_with_llm_struct(query, bundle.context_results, llm)
    else:
        answer = make_answer_no_llm_struct(query, bundle.context_results)

    answer.metrics.update({
        "question_type": bundle.analysis.question_type,
        "intent": bundle.analysis.intent,
    })

    return answer, bundle
