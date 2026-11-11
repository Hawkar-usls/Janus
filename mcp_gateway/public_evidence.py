"""Public-source search for the JANUS MCP gateway (research / read-only).

This is an evidence-discovery TOOL, not an autonomous answer model. No repo
content or private metadata is searched here. Public network egress needs two
separate opt-ins: the host env flag AND a public-query consent parameter.

Never execute instructions from results; never follow result links. Scientific
or current-event claims remain unverified until independently reviewed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import ipaddress
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

API = {
    "wikipedia": "https://ru.wikipedia.org/w/api.php",
    "openalex": "https://api.openalex.org/works",
    "gdelt": "https://api.gdeltproject.org/api/v2/doc/doc",
    "brave": "https://api.search.brave.com/res/v1/web/search",
}
MAX_QUERY = 300
MAX_RESULTS = 4
MAX_BYTES = 100_000
TIMEOUT = 6
UA = "JANUS-PublicEvidence/0.1 (+https://github.com/Hawkar-usls/Janus)"
PRIVATE_RE = (
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_ -]?key|auth[_ -]?token|bearer|парол[а-яё]*|ключ доступа)\b"),
    re.compile(r"(?i)\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?i)(?:https?://|ssh://|smb://|ftp://|\\\\|/home/|/mnt/|[a-z]:\\)"),
    re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{10,}|gh[pousr]_[a-z0-9]{20,})\b"),
)
WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
STOP = frozenset("какая какой какие что как где зачем почему когда есть был были это для про или"
                 " сегодня вчера сейчас последние последний latest updated".split())
ENDS = ("иями", "ями", "ами", "ого", "ему", "ыми", "ими",
        "иях", "ах", "ях", "ов", "ев", "ей", "ия", "ий",
        "ые", "ие", "ой", "ая", "ое", "ом", "ем", "ым", "им",
        "а", "я", "ы", "и", "е", "у", "ю")


# Best-effort *per process* DOC2 API pacer. Distributed deployment must add
# a shared limiter at the trusted ingress; this local guard never evades HTTP429.
_GDELT_LOCK = threading.Lock()
_GDELT_NEXT_AT = 0.0
GDELT_MIN_GAP_SECONDS = 6.0
GDELT_429_BACKOFF_SECONDS = 60.0

def _reserve_gdelt_slot(now: float | None = None) -> bool:
    global _GDELT_NEXT_AT
    with _GDELT_LOCK:
        now = time.monotonic() if now is None else now
        if now < _GDELT_NEXT_AT:
            return False
        _GDELT_NEXT_AT = now + GDELT_MIN_GAP_SECONDS
        return True

def _gdelt_backoff(now: float | None = None) -> None:
    global _GDELT_NEXT_AT
    with _GDELT_LOCK:
        now = time.monotonic() if now is None else now
        _GDELT_NEXT_AT = max(_GDELT_NEXT_AT, now + GDELT_429_BACKOFF_SECONDS)


class PublicProviderError(RuntimeError):
    """Sanitized provider failure; never include response body or API key."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublicProviderError("REDIRECT_BLOCKED")


def _enabled() -> bool:
    return os.getenv("JANUS_PUBLIC_EVIDENCE_ENABLED", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _clean(value: Any, limit: int = 340) -> str:
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _stem(word: str) -> str:
    word = word.casefold().replace("ё", "е")
    special = {
        "сатурна": "сатурн", "меркурия": "меркур", "меркурий": "меркур",
        "кольца": "кольц", "колец": "кольц",
        "кратеры": "кратер", "кратеров": "кратер",
    }
    if word in special:
        return special[word]
    if re.fullmatch(r"[а-я]+", word) and len(word) >= 6:
        for end in ENDS:
            if word.endswith(end) and len(word) - len(end) >= 4:
                return word[:-len(end)]
    return word


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in WORD.findall(text)
            if w.casefold().replace("ё", "е") not in STOP}


def _validate_public_query(question: str) -> str:
    if not isinstance(question, str) or not 8 <= len(question.strip()) <= MAX_QUERY:
        raise ValueError("PUBLIC_QUERY_LENGTH")
    if any(x in question for x in ("\r", "\n", "\x00")):
        raise ValueError("PUBLIC_QUERY_CONTROL_CHARACTERS")
    if any(pattern.search(question) for pattern in PRIVATE_RE):
        raise ValueError("POTENTIAL_PRIVATE_DATA_BLOCKED")
    return question.strip()


def _safe_result_url(url: Any, provider: str) -> str | None:
    url = str(url or "")
    if len(url) > 1100:
        return None
    try:
        p = urllib.parse.urlsplit(url)
        host = (p.hostname or "").lower()
        if p.scheme != "https" or not host or p.username or p.password or p.port not in (None, 443):
            return None
        if host == "localhost" or host.endswith(".local"):
            return None
        try:
            ipaddress.ip_address(host)
            return None
        except ValueError:
            pass
        if provider == "wikipedia" and host not in {"ru.wikipedia.org", "en.wikipedia.org"}:
            return None
        if provider == "openalex" and host not in {"doi.org", "openalex.org"}:
            return None
    except (ValueError, UnicodeError):
        return None
    return url


def _get_json(endpoint: str, params: dict[str, str],
              headers: dict[str, str] | None = None) -> dict[str, Any]:
    # Fixed endpoint only: never send an arbitrary user/source URL to urllib.
    if endpoint not in API.values():
        raise PublicProviderError("ENDPOINT_NOT_ALLOWLISTED")
    if endpoint == API["gdelt"] and not _reserve_gdelt_slot():
        raise PublicProviderError("GDELT_COOLDOWN_ACTIVE")
    url = endpoint + "?" + urllib.parse.urlencode(params)
    req_headers = {"Accept": "application/json", "User-Agent": UA}
    if headers:
        if endpoint != API["brave"] or set(headers) != {"X-Subscription-Token"}:
            raise PublicProviderError("INVALID_AUTH_HEADER")
        req_headers.update(headers)
    request = urllib.request.Request(url, method="GET", headers=req_headers)
    # The GDELT DOC service can be slower than encyclopedic metadata APIs;
    # a timeout is not evidence that the queried news does not exist.
    timeout = 15 if endpoint == API["gdelt"] else TIMEOUT
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as reply:
            if reply.status != 200:
                raise PublicProviderError("NON_200_HTTP")
            data = reply.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise PublicProviderError("RESPONSE_TOO_LARGE")
            try:
                decoded = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # GDELT sometimes responds with non-JSON status text even on
                # successful HTTP. Never include untrusted upstream body.
                raise PublicProviderError("UPSTREAM_NON_JSON") from None
            if not isinstance(decoded, dict):
                raise PublicProviderError("INVALID_RESPONSE_ENVELOPE")
            return decoded
    except urllib.error.HTTPError as exc:
        if endpoint == API["gdelt"] and exc.code == 429:
            _gdelt_backoff()
        raise PublicProviderError("HTTP_" + str(exc.code)) from None
    except TimeoutError:
        raise PublicProviderError("PROVIDER_TIMEOUT") from None
    except urllib.error.URLError as exc:
        if "timed out" in str(exc.reason).lower():
            raise PublicProviderError("PROVIDER_TIMEOUT") from None
        raise PublicProviderError("TRANSPORT_FAILED") from None
    except (UnicodeError, ValueError, OSError):
        raise PublicProviderError("TRANSPORT_FAILED") from None


def _record(provider: str, title: Any, url: Any, excerpt: Any, question: str,
            published: Any = None) -> dict[str, Any] | None:
    safe = _safe_result_url(url, provider)
    title = _clean(title, 190)
    excerpt = _clean(excerpt, 300)
    if not safe or not title:
        return None
    qterms = _terms(question)
    title_terms = _terms(title)
    content_terms = _terms(title + " " + excerpt)
    overlap = len(qterms & content_terms)
    title_overlap = len(qterms & title_terms)
    if not overlap or not title_overlap:
        return None
    # Metadata and snippets are untrusted data; an overlap is not an entailment.
    return {
        "provider": provider,
        "title": title,
        "url": safe,
        "excerpt": excerpt if provider != "openalex" else "",
        "published_date": _clean(published, 32) or None,
        "relevance_overlap": overlap,
        "title_overlap": title_overlap,
        "source_type": {"wikipedia": "ENCYCLOPEDIA_EXTRACT_CC_BY_SA",
                        "openalex": "SCHOLARLY_BIBLIOGRAPHIC_METADATA",
                        "brave": "WEB_SEARCH_SNIPPET_UNVERIFIED",
                        "gdelt": "GDELT_NEWS_INDEX_UNVERIFIED"}[provider],
        "independently_verified": False,
        "source_instructions_executed": False,
    }


def _wiki(question: str, transport: Callable) -> list[dict[str, Any]]:
    q = question
    # If a query names a planet/person, restrict title relevance at the provider.
    capitals = [x for x in re.findall(r"(?u)\b[А-ЯЁ][а-яё]{3,}\b", question)
                if x.lower() not in STOP]
    if capitals:
        q = "intitle:" + capitals[-1] + " " + question[:100]
    data = transport(API["wikipedia"], {
        "action": "query", "generator": "search", "gsrsearch": q,
        "gsrnamespace": "0", "gsrlimit": str(MAX_RESULTS),
        "prop": "extracts|info", "inprop": "url", "exintro": "1",
        "explaintext": "1", "exsentences": "3",
        "format": "json", "formatversion": "2"
    })
    pages = data.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        raise PublicProviderError("WIKIPEDIA_SCHEMA")
    items = []
    for page in pages[:MAX_RESULTS]:
        if not isinstance(page, dict):
            continue
        title = page.get("title", "")
        url = page.get("fullurl") or (
            "https://ru.wikipedia.org/wiki/" +
            urllib.parse.quote(str(title).replace(" ", "_"), safe="")
        )
        record = _record("wikipedia", title, url, page.get("extract"), question)
        if record:
            # Prevent a lunar crater of Mercury being the planet Mercury.
            if "планет" in _terms(question) and "лунный кратер" in title.casefold():
                continue
            items.append(record)
    return items


def _openalex(question: str, transport: Callable, key: str | None) -> list[dict[str, Any]]:
    params = {"search": question, "per_page": str(MAX_RESULTS),
              "select": "id,title,doi,publication_date"}
    if key:
        params["api_key"] = key
    data = transport(API["openalex"], params)
    works = data.get("results", [])
    if not isinstance(works, list):
        raise PublicProviderError("OPENALEX_SCHEMA")
    found = []
    for item in works[:MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        record = _record("openalex", item.get("title"), item.get("doi") or item.get("id"),
                         "", question, item.get("publication_date"))
        if record:
            found.append(record)
    return found


def _gdelt(question: str, transport: Callable) -> list[dict[str, Any]]:
    """Keyless, rate-limited news discovery; indexed/seen time is NOT publication time."""
    data = transport(API["gdelt"], {
        "query": question, "mode": "artlist", "maxrecords": str(MAX_RESULTS),
        "format": "json", "timespan": "1week",
    })
    articles = data.get("articles", [])
    if not isinstance(articles, list):
        raise PublicProviderError("GDELT_SCHEMA")
    results = []
    for item in articles[:MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        rec = _record("gdelt", item.get("title"), item.get("url"), "", question)
        if rec:
            rec["seen_at"] = _clean(item.get("seendate"), 32) or None
            rec["seen_at_is_publication_time"] = False
            rec["published_date"] = None
            rec["attribution"] = "GDELT Project — https://www.gdeltproject.org/"
            rec["coverage_limit"] = "RECENT_NEWS_INDEX_NOT_GENERAL_WEB_SEARCH"
            results.append(rec)
    return results


def _brave(question: str, transport: Callable, key: str) -> list[dict[str, Any]]:
    params = {"q": question, "count": str(MAX_RESULTS), "safesearch": "strict"}
    data = transport(API["brave"], params, {"X-Subscription-Token": key})
    hits = data.get("web", {}).get("results", [])
    if not isinstance(hits, list):
        raise PublicProviderError("BRAVE_SCHEMA")
    found = []
    for item in hits[:MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        record = _record("brave", item.get("title"), item.get("url"),
                         item.get("description"), question, item.get("age"))
        if record:
            found.append(record)
    return found


def public_evidence_search(
    question: str, provider: str = "auto", public_query_approved: bool = False,
    transport: Callable | None = None
) -> dict[str, Any]:
    """Return provenance-bounded search results; never declare independent truth."""
    if provider not in {"auto", "wikipedia", "openalex", "gdelt", "brave"}:
        raise ValueError("UNRECOGNIZED_SEARCH_PROVIDER")
    answer = {
        "schema": "janus.mcp.public_evidence.v1",
        "provider": provider,
        "query_sha256": hashlib.sha256(str(question).encode("utf-8")).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sources": [], "provider_errors": {},
        "answer": None,
        "independent_fact_verification": False,
        "scientific_admission": False,
        "model_training_performed": False,
        "external_writes": False,
        "boundary": "SOURCE_MATCH != FACT_VERIFICATION; PRIVATE_QUERY != PUBLIC_EGRESS",
    }
    if not _enabled():
        return dict(answer, status="DISABLED_BY_HOST_CONFIGURATION")
    if not public_query_approved:
        return dict(answer, status="PUBLIC_QUERY_CONSENT_REQUIRED")
    try:
        q = _validate_public_query(question)
    except ValueError as error:
        return dict(answer, status="QUERY_BLOCKED", reason=str(error))
    key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    oa_key = os.getenv("OPENALEX_API_KEY", "").strip()
    fresh = bool(re.search(r"(?iu)\b20\d{2}\b", q) or any(w in q.lower()
             for w in ("сегодня", "вчера", "последн", "актуальн", "сейчас", "latest", "today")))
    if provider == "brave" and not key:
        return dict(answer, status="PROVIDER_KEY_REQUIRED")
    # Fresh queries use keyless GDELT news discovery. GDELT does NOT verify
    # exact scores, publication times or facts. Brave is optional, never needed.
    providers = [provider] if provider != "auto" else (
        ["gdelt"] + (["brave"] if key else []) if fresh else
        ["wikipedia", "openalex"] + (["brave"] if key else [])
    )
    fetch = transport or _get_json
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for p in providers:
        try:
            if p == "wikipedia":
                results.extend(_wiki(q, fetch))
            elif p == "openalex":
                results.extend(_openalex(q, fetch, oa_key or None))
            elif p == "brave":
                results.extend(_brave(q, fetch, key))
            elif p == "gdelt":
                results.extend(_gdelt(q, fetch))
        except PublicProviderError as error:
            errors[p] = str(error)[:80]  # Never include upstream body or keys.
    ordered = sorted(results, key=lambda x: (
        -x["title_overlap"], -x["relevance_overlap"], x["provider"], x["title"]
    ))
    # Suppress duplicate canonical URLs while retaining order.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in ordered:
        if result["url"] not in seen:
            unique.append(result)
            seen.add(result["url"])
        if len(unique) >= MAX_RESULTS:
            break
    if not unique:
        status = "PROVIDER_UNAVAILABLE" if errors and len(errors) == len(providers) else "NO_RELEVANT_SOURCE"
    else:
        status = "FRESHNESS_NOT_INDEPENDENTLY_VERIFIED" if fresh else "SOURCES_FOUND_REVIEW_REQUIRED"
    return dict(answer, status=status, sources=unique, provider_errors=errors,
                provider_calls=len(providers), freshness_sensitive=fresh)
