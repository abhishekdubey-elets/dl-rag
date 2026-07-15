"""Heuristic query understanding.

:class:`HeuristicQueryAnalyzer` turns a raw user query into a structured
:class:`QueryAnalysis` using only deterministic rules — keyword/intent tables,
the curated gazetteer, and light regex for years and comparisons. An optional
:class:`LLMClient` may be injected by the integrator for future augmentation,
but the analyzer is fully functional (and deterministic) without it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from dl_rag.config import Settings
from dl_rag.constants import GAZETTEER, INDIAN_STATES, INTENT_KEYWORDS
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import QueryAnalysis, TimeRange
from dl_rag.models.enums import ContentType, QueryType
from dl_rag.protocols import LLMClient
from dl_rag.utils.text import clean_whitespace, extract_years

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_YEAR = r"(19[89]\d|20[0-4]\d)"
# 2+ consecutive Capitalized tokens → candidate proper-noun entity.
_PROPER_PHRASE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'.\-]*(?:\s+[A-Z][A-Za-z0-9&'.\-]*)+)"
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "was", "were", "be", "by", "with", "as", "at", "it", "its",
        "this", "that", "these", "those", "what", "which", "who", "how", "why",
        "when", "where", "from", "about", "into", "over", "than", "do", "does",
        "did", "has", "have", "had", "can", "will", "would", "should", "could",
        "me", "my", "you", "your", "we", "our", "they", "their", "i",
    }
)

# Sentence-initial words that get capitalized and must not seed a proper-noun
# entity candidate (verbs / wh-words / articles / prepositions).
_LEADING_NOISE: frozenset[str] = frozenset(
    {
        "compare", "comparison", "what", "who", "how", "when", "where", "why",
        "which", "list", "show", "tell", "give", "find", "explain", "describe",
        "summarize", "summarise", "is", "are", "was", "were", "do", "does",
        "did", "can", "could", "should", "would", "the", "a", "an", "in", "on",
        "of", "for", "about", "please", "top", "best", "recommend", "suggest",
    }
)

# Function words trimmed from either end of a derived topic base.
_EDGE_FUNC: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "in", "on", "for", "over", "to", "and", "with",
     "by", "at", "has", "have", "how"}
)

# Queries about the present/future — retrieval should favour fresh content.
_RECENCY_RE = re.compile(
    r"\b(next|upcoming|latest|newest|current(?:ly)?|recent(?:ly)?|"
    r"this (?:year|month)|right now|nowadays|today)\b"
)

# Temporal phrases stripped when deriving a timeline/trend topic base.
_TEMPORAL_STRIP: tuple[str, ...] = (
    "timeline of", "timeline", "evolution of", "evolution", "history of",
    "over the years", "over time", "chronology of", "chronology",
    "developments in", "developments", "trend of", "trends in", "trends",
    "trend", "growth of", "growth", "how has", "how have", "changed over",
    "change over", "shift in", "since", "emerging",
)


class HeuristicQueryAnalyzer:
    """Deterministic, rule-based implementation of the ``QueryAnalyzer`` Protocol."""

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self._settings = settings
        self._llm = llm  # reserved for optional augmentation; not required.

    async def analyze(self, query: str) -> QueryAnalysis:
        normalized = clean_whitespace(query)
        lowered = normalized.lower()

        query_type = self._classify(lowered)
        entities = self._extract_entities(normalized, lowered)
        time_range = self._extract_time_range(lowered)
        content_type_filter = self._content_types(lowered, query_type)
        sub_queries = self._sub_queries(query_type, normalized, lowered)
        keywords = self._keywords(lowered)

        reasoning = (
            f"heuristic classify -> {query_type.value}; "
            f"entities={len(entities)}; "
            f"time={'set' if time_range.is_set() else 'none'}; "
            f"content_types={[c.value for c in content_type_filter]}; "
            f"sub_queries={len(sub_queries)}"
        )

        logger.debug(
            "retrieval.query_understanding.done",
            query_type=query_type.value,
            entities=len(entities),
            time_set=time_range.is_set(),
            sub_queries=len(sub_queries),
        )

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
            time_range=time_range,
            content_type_filter=content_type_filter,
            sub_queries=sub_queries,
            recency_sensitive=bool(_RECENCY_RE.search(lowered)),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _classify(self, lowered: str) -> QueryType:
        if not lowered:
            return QueryType.GENERAL

        primary: list[tuple[QueryType, list[str]]] = [
            (
                QueryType.COMPARISON,
                [*INTENT_KEYWORDS["comparison"], "difference between",
                 "compared to", "compared with"],
            ),
            (QueryType.TIMELINE, list(INTENT_KEYWORDS["timeline"])),
            (QueryType.TREND, list(INTENT_KEYWORDS["trend"])),
            (QueryType.DEFINITION, list(INTENT_KEYWORDS["definition"])),
            (QueryType.RANKING, list(INTENT_KEYWORDS["ranking"])),
            (QueryType.INTERVIEW, list(INTENT_KEYWORDS["interview"])),
            (QueryType.PERSON, list(INTENT_KEYWORDS["person"])),
            (QueryType.STATISTICS, list(INTENT_KEYWORDS["statistics"])),
            (QueryType.RECOMMENDATION, list(INTENT_KEYWORDS["recommendation"])),
        ]
        for qtype, phrases in primary:
            if _contains_any(lowered, phrases):
                return qtype

        extended: list[tuple[QueryType, list[str]]] = [
            (
                QueryType.SUMMARIZATION,
                ["summarize", "summarise", "summary", "overview", "recap",
                 "tl;dr", "in brief", "key points", "key takeaways"],
            ),
            (QueryType.MAGAZINE,
             ["magazine", "issue", "edition", "cover story", "special story"]),
            (QueryType.EVENT,
             ["event", "summit", "conference", "webinar", "expo", "awards",
              "conclave", "symposium", "seminar"]),
            (QueryType.POLICY,
             ["policy", "policies", "scheme", "regulation", "guidelines",
              "notification", "mandate", "circular", "amendment"]),
            (QueryType.INSTITUTION,
             ["university", "universities", "college", "institute", "campus",
              "iit", "iim", "nit"]),
        ]
        for qtype, phrases in extended:
            if _contains_any(lowered, phrases):
                return qtype

        return QueryType.GENERAL

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #
    def _extract_entities(self, normalized: str, lowered: str) -> list[str]:
        found: list[tuple[int, str]] = []
        seen_lower: set[str] = set()

        # 1) Curated gazetteer (canonical + aliases).
        for canonical, (_etype, aliases) in GAZETTEER.items():
            best_pos: int | None = None
            matched_surfaces: list[str] = []
            for surface in (canonical, *aliases):
                pos = _phrase_pos(lowered, surface.lower())
                if pos is not None:
                    matched_surfaces.append(surface.lower())
                    best_pos = pos if best_pos is None else min(best_pos, pos)
            if best_pos is not None and canonical.lower() not in seen_lower:
                found.append((best_pos, canonical))
                seen_lower.add(canonical.lower())
                seen_lower.update(matched_surfaces)

        # 2) Indian states / UTs (canonical == the state name).
        for state in INDIAN_STATES:
            pos = _phrase_pos(lowered, state.lower())
            if pos is not None and state.lower() not in seen_lower:
                found.append((pos, state))
                seen_lower.add(state.lower())

        # 3) Salient Capitalized multi-word phrases as candidate entities. Drop
        #    sentence-initial noise ("Compare NEP ..." -> "NEP", already gazette).
        for match in _PROPER_PHRASE.finditer(normalized):
            words = match.group(1).split()
            while words and words[0].lower() in _LEADING_NOISE:
                words.pop(0)
            if len(words) < 2:
                continue
            phrase = " ".join(words).strip(" -.'")
            low = phrase.lower()
            if not phrase or low in seen_lower:
                continue
            found.append((match.start(), phrase))
            seen_lower.add(low)

        found.sort(key=lambda item: item[0])
        return [name for _, name in found]

    # ------------------------------------------------------------------ #
    # Time range
    # ------------------------------------------------------------------ #
    def _extract_time_range(self, lowered: str) -> TimeRange:
        # between A and B
        m = re.search(rf"\bbetween\s+{_YEAR}\s+and\s+{_YEAR}\b", lowered)
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            return TimeRange(from_year=a, to_year=b)

        # A <connector> B  (e.g. "2010 to 2020", "2010-2020")
        m = re.search(
            rf"\b{_YEAR}\s*(?:-|–|—|to|through|until|till)\s*{_YEAR}\b", lowered
        )
        if m:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            return TimeRange(from_year=a, to_year=b)

        # last / past N years
        m = re.search(r"\b(?:last|past|previous)\s+(\d{1,2})\s+years?\b", lowered)
        if m:
            n = int(m.group(1))
            current = self._current_year()
            return TimeRange(from_year=current - n, to_year=current)

        # since / from / after A  -> open-ended lower bound
        m = re.search(rf"\b(?:since|from|after)\s+{_YEAR}\b", lowered)
        if m:
            return TimeRange(from_year=int(m.group(1)), to_year=None)

        # before / prior to / until A -> open-ended upper bound
        m = re.search(rf"\b(?:before|prior to|until|till|up to)\s+{_YEAR}\b", lowered)
        if m:
            return TimeRange(from_year=None, to_year=int(m.group(1)))

        # bare years: two+ distinct -> span; single -> point (covers "in YYYY")
        distinct = list(dict.fromkeys(extract_years(lowered)))
        if len(distinct) >= 2:
            return TimeRange(from_year=min(distinct), to_year=max(distinct))
        if len(distinct) == 1:
            year = distinct[0]
            return TimeRange(from_year=year, to_year=year)

        return TimeRange()

    def _current_year(self) -> int:
        override = getattr(self._settings, "current_year", None)
        if isinstance(override, int) and override > 0:
            return override
        from datetime import date

        return date.today().year

    # ------------------------------------------------------------------ #
    # Content-type filter
    # ------------------------------------------------------------------ #
    def _content_types(self, lowered: str, qtype: QueryType) -> list[ContentType]:
        result: list[ContentType] = []

        def add(*cts: ContentType) -> None:
            for ct in cts:
                if ct not in result:
                    result.append(ct)

        if _contains_any(lowered, ["interview", "interviews", "in conversation"]) \
                or qtype == QueryType.INTERVIEW:
            add(ContentType.INTERVIEW)
        if _contains_any(lowered, ["ranking", "rankings", "rank", "nirf"]) \
                or qtype == QueryType.RANKING:
            add(ContentType.RANKING)
        if _contains_any(lowered, ["policy", "policies"]) \
                or qtype == QueryType.POLICY:
            add(ContentType.POLICY)
        if _contains_any(lowered, ["magazine", "issue", "edition"]) \
                or qtype == QueryType.MAGAZINE:
            add(ContentType.MAGAZINE_ISSUE, ContentType.MAGAZINE_ARTICLE)
        if _contains_any(
            lowered,
            ["video", "videos", "watch", "recording", "recordings",
             "youtube", "footage", "session video"],
        ):
            # An explicit video ask narrows to video content only — mixing in
            # article types here would bury the links the user asked for.
            return [ContentType.VIDEO]

        return result

    # ------------------------------------------------------------------ #
    # Sub-queries
    # ------------------------------------------------------------------ #
    def _sub_queries(
        self, qtype: QueryType, normalized: str, lowered: str
    ) -> list[str]:
        if qtype == QueryType.COMPARISON:
            return self._comparison_sub_queries(normalized)
        if qtype in (QueryType.TIMELINE, QueryType.TREND):
            return self._temporal_sub_queries(normalized, lowered)
        return []

    def _comparison_sub_queries(self, normalized: str) -> list[str]:
        # Prefer an explicit "A vs B" / "A versus B" split.
        parts = re.split(r"\s+(?:vs\.?|versus)\s+", normalized, flags=re.IGNORECASE)
        if len(parts) < 2:
            stripped = re.sub(
                r"^\s*(?:compare|comparison of|comparison between|"
                r"difference between|differences between|compared)\b[:\s]*",
                "",
                normalized,
                flags=re.IGNORECASE,
            )
            parts = re.split(
                r"\s+(?:and|with|versus|vs\.?)\s+", stripped, flags=re.IGNORECASE
            )

        subjects: list[str] = []
        for part in parts:
            cleaned = clean_whitespace(part).strip(" ?.,:;-")
            if cleaned:
                subjects.append(cleaned)
            if len(subjects) == 2:
                break
        return subjects if len(subjects) == 2 else []

    def _temporal_sub_queries(self, normalized: str, lowered: str) -> list[str]:
        base = self._topic_base(normalized)
        years = list(dict.fromkeys(extract_years(lowered)))
        if years:
            return [f"{base} in {year}".strip() for year in years]
        # No explicit years → decompose into era buckets.
        return [
            f"{base} before 2015".strip(),
            f"{base} 2015 to 2020".strip(),
            f"{base} since 2020".strip(),
        ]

    @staticmethod
    def _topic_base(normalized: str) -> str:
        base = re.sub(_YEAR, " ", normalized)
        base = re.sub(
            r"\b(?:last|past|previous)\s+\d{1,2}\s+years?\b", " ", base,
            flags=re.IGNORECASE,
        )
        for phrase in _TEMPORAL_STRIP:
            base = re.sub(rf"\b{re.escape(phrase)}\b", " ", base, flags=re.IGNORECASE)
        base = clean_whitespace(base).strip(" ?.,:;-")
        # Trim dangling function words left at either end after stripping.
        words = base.split()
        while words and words[0].lower() in _EDGE_FUNC:
            words.pop(0)
        while words and words[-1].lower() in _EDGE_FUNC:
            words.pop()
        base = " ".join(words)
        return base or clean_whitespace(normalized)

    # ------------------------------------------------------------------ #
    # Keywords
    # ------------------------------------------------------------------ #
    @staticmethod
    def _keywords(lowered: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for token in _WORD.findall(lowered):
            if len(token) <= 2 or token in _STOPWORDS or token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= 12:
                break
        return out


# --------------------------------------------------------------------------- #
# Module-level matching helpers
# --------------------------------------------------------------------------- #
def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    """True if any phrase occurs in ``text`` as a whole word/phrase."""
    for phrase in phrases:
        needle = phrase.strip().lower()
        if needle and re.search(r"\b" + re.escape(needle) + r"\b", text):
            return True
    return False


def _phrase_pos(text: str, phrase: str) -> int | None:
    """Start index of ``phrase`` in ``text`` as a whole word/phrase, else None."""
    phrase = phrase.strip()
    if not phrase:
        return None
    match = re.search(r"\b" + re.escape(phrase) + r"\b", text)
    return match.start() if match else None
