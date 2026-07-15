"""Deterministic gazetteer entity extraction, augmented by optional spaCy NER.

Precision-first: the curated :data:`GAZETTEER` and :data:`INDIAN_STATES` give
high-confidence, canonicalised entities; statistical NER (spaCy, lazily loaded
and fully guarded) catches the long tail. Also extracts lightweight keywords and
subject-predicate-object relations triggered by :data:`RELATION_TRIGGERS`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from dl_rag.config import Settings
from dl_rag.constants import GAZETTEER, INDIAN_STATES, RELATION_TRIGGERS
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import Entity, Relation, SourceDocument
from dl_rag.models.enums import EntityType, RelationType
from dl_rag.utils.text import split_sentences

logger = get_logger(__name__)

_CAP_SEQ_RE = re.compile(
    r"[A-Z][A-Za-z0-9&.\-']*(?:\s+[A-Z][A-Za-z0-9&.\-']*){0,2}"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

# Honorific-prefixed names → PERSON. Statistical NER (en_core_web_sm) misses
# many Indian names, but event/speaker coverage almost always uses titles:
# "Prof. Anil Sahasrabudhe", "Shri Dharmendra Pradhan", "Dr. K. Kasturirangan".
_HONORIFIC_PERSON_RE = re.compile(
    r"(?:Prof(?:essor)?\.?|Dr\.?|Shri|Smt\.?|Mr\.?|Mrs\.?|Ms\.?|"
    r"Hon(?:'|’)?ble|Justice|Padma\s+Shri)\s+"
    r"([A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,3})"
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

# Job titles / generic nouns spaCy routinely mislabels as PERSON, and that make
# meaningless spoke_at objects. Matched against the full normalized name.
_ROLE_WORDS: frozenset[str] = frozenset(
    {
        "principal", "chancellor", "vice chancellor", "director", "secretary",
        "minister", "professor", "dean", "registrar", "ceo", "cfo", "cto",
        "coo", "cio", "founder", "co-founder", "president", "chairman",
        "chairperson", "chairwoman", "head", "officer", "manager", "chief",
        "schools", "school", "learning", "education", "student", "students",
        "teacher", "teachers", "university", "college", "government", "sir",
        "madam", "guest", "speaker", "delegates", "jee", "neet", "cbse",
    }
)


def _plausible_person(name: str, *, from_honorific: bool) -> bool:
    """Filter spaCy PERSON noise: role words, acronyms, fragments, bare surnames."""
    normalized = Entity.normalize(name)
    if normalized in _ROLE_WORDS or normalized in _STOPWORDS:
        return False
    if not name[0].isalpha():  # "-Learning" style fragments
        return False
    if name.isupper():  # acronyms (JEE, NEET) are never people
        return False
    # A bare single token ("Kumar", "Wong") is too ambiguous from NER alone,
    # but fine when an honorific vouches for it ("Dr. Kasturirangan").
    if len(name.split()) < 2 and not from_honorific:
        return False
    return True

# Small country lexicon so spaCy GPEs resolve to COUNTRY vs CITY sensibly.
_COUNTRIES = {
    "india", "united states", "usa", "u.s.", "u.s.a.", "america",
    "united kingdom", "uk", "britain", "china", "japan", "australia",
    "canada", "germany", "france", "singapore", "bangladesh", "nepal",
    "sri lanka", "pakistan", "bhutan", "russia", "brazil", "south africa",
    "israel", "finland", "south korea", "new zealand", "netherlands",
}

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
        "of", "on", "in", "to", "with", "as", "by", "at", "from", "into",
        "is", "are", "was", "were", "be", "been", "being", "this", "that",
        "these", "those", "it", "its", "he", "she", "they", "them", "his",
        "her", "their", "we", "you", "our", "your", "i", "which", "who",
        "whom", "whose", "what", "when", "where", "why", "how", "will",
        "would", "can", "could", "should", "may", "might", "must", "shall",
        "have", "has", "had", "do", "does", "did", "not", "no", "yes",
        "also", "more", "most", "such", "some", "any", "all", "than",
        "there", "here", "over", "under", "after", "before", "while",
        "about", "against", "between", "during", "through", "per", "via",
        "said", "says", "say", "one", "two", "new", "many", "much", "so",
        "up", "out", "down", "off", "only", "own", "same", "other", "each",
        "both", "few", "now", "just", "very", "too", "s", "t", "mr", "mrs",
        "dr", "ms",
    }
)

_MAX_NER_CHARS = 200_000


def _compile_terms(terms: list[str]) -> re.Pattern[str] | None:
    """Whole-word, case-insensitive alternation over ``terms`` (longest first)."""
    escaped = sorted({re.escape(t) for t in terms if t}, key=len, reverse=True)
    if not escaped:
        return None
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


class EntityExtractor:
    """Extract canonical entities, keywords, and relations from text."""

    def __init__(self, settings: Settings, use_spacy: bool = True) -> None:
        self._settings = settings
        self._use_spacy = use_spacy
        self._nlp: Any | None = None
        self._spacy_attempted = False
        self._spacy_failed = False
        self._patterns = self._build_patterns()

    # ------------------------------------------------------------------ #
    # Pattern table
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_patterns() -> list[tuple[str, EntityType, list[str], re.Pattern[str]]]:
        patterns: list[tuple[str, EntityType, list[str], re.Pattern[str]]] = []
        for canonical, (etype, aliases) in GAZETTEER.items():
            compiled = _compile_terms([canonical, *aliases])
            if compiled is not None:
                patterns.append((canonical, etype, list(aliases), compiled))
        for state in INDIAN_STATES:
            compiled = _compile_terms([state])
            if compiled is not None:
                patterns.append((state, EntityType.STATE, [], compiled))
        return patterns

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(
        self, text: str, source_url: str = ""
    ) -> tuple[list[Entity], list[str], list[Relation]]:
        text = text or ""
        entities = self._match_gazetteer(text)
        self._augment_spacy(text, entities)
        self._augment_honorific_persons(text, entities)
        entity_list = list(entities.values())
        keywords = self._keywords(text)
        relations = self._relations(text, entity_list, source_url)
        return entity_list, keywords, relations

    def extract_from_document(
        self, doc: SourceDocument
    ) -> tuple[list[Entity], list[str], list[Relation]]:
        text = f"{doc.title}\n\n{doc.content_markdown}"
        return self.extract(text, source_url=doc.url)

    # ------------------------------------------------------------------ #
    # Gazetteer matching
    # ------------------------------------------------------------------ #
    def _match_gazetteer(self, text: str) -> dict[str, Entity]:
        found: dict[str, Entity] = {}
        for name, etype, aliases, pattern in self._patterns:
            count = len(pattern.findall(text))
            if count == 0:
                continue
            normalized = Entity.normalize(name)
            existing = found.get(normalized)
            if existing is not None:
                existing.mention_count += count
                continue
            found[normalized] = Entity(
                id=Entity.make_id(name),
                name=name,
                normalized_name=normalized,
                type=etype,
                aliases=list(aliases),
                mention_count=count,
            )
        return found

    # ------------------------------------------------------------------ #
    # Optional spaCy augmentation
    # ------------------------------------------------------------------ #
    def _load_spacy(self) -> Any | None:
        if not self._use_spacy or self._spacy_failed:
            return None
        if self._nlp is not None:
            return self._nlp
        if self._spacy_attempted:
            return None
        self._spacy_attempted = True
        try:
            import spacy
        except ImportError:
            logger.warning("entities.spacy_unavailable", reason="import_error")
            self._spacy_failed = True
            return None
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except Exception as exc:  # noqa: BLE001 - model missing/incompatible
            logger.warning("entities.spacy_model_unavailable", error=str(exc))
            self._spacy_failed = True
            return None
        return self._nlp

    @staticmethod
    def _map_label(label: str, text: str) -> EntityType | None:
        if label == "PERSON":
            return EntityType.PERSON
        if label == "ORG":
            return EntityType.ORGANIZATION
        if label == "GPE":
            low = text.strip().lower()
            if low in _COUNTRIES:
                return EntityType.COUNTRY
            if any(low == s.lower() for s in INDIAN_STATES):
                return EntityType.STATE
            return EntityType.CITY
        return None

    def _augment_spacy(self, text: str, entities: dict[str, Entity]) -> None:
        nlp = self._load_spacy()
        if nlp is None or not text.strip():
            return
        try:
            doc = nlp(text[:_MAX_NER_CHARS])
        except Exception as exc:  # noqa: BLE001 - never let NER crash extraction
            logger.warning("entities.spacy_run_failed", error=str(exc))
            return

        counts: Counter[str] = Counter()
        info: dict[str, tuple[str, EntityType]] = {}
        for ent in getattr(doc, "ents", []):
            etype = self._map_label(getattr(ent, "label_", ""), ent.text)
            if etype is None:
                continue
            name = " ".join(ent.text.split()).strip(" .,:;\"'")
            name = _LEADING_ARTICLE_RE.sub("", name)  # "the World…" → "World…"
            if len(name) < 2 or not any(ch.isalpha() for ch in name):
                continue
            normalized = Entity.normalize(name)
            if normalized in _STOPWORDS:
                continue
            if etype == EntityType.PERSON and not _plausible_person(
                name, from_honorific=False
            ):
                continue
            counts[normalized] += 1
            info.setdefault(normalized, (name, etype))

        for normalized, count in counts.items():
            existing = entities.get(normalized)
            if existing is not None:
                existing.mention_count += count
                continue
            name, etype = info[normalized]
            entities[normalized] = Entity(
                id=Entity.make_id(name),
                name=name,
                normalized_name=normalized,
                type=etype,
                mention_count=count,
            )

    def _augment_honorific_persons(
        self, text: str, entities: dict[str, Entity]
    ) -> None:
        """Add PERSON entities from honorific-prefixed names (deterministic)."""
        counts: Counter[str] = Counter()
        names: dict[str, str] = {}
        for match in _HONORIFIC_PERSON_RE.finditer(text):
            name = " ".join(match.group(1).split()).strip(" .,:;\"'")
            if len(name) < 3 or not _plausible_person(name, from_honorific=True):
                continue
            normalized = Entity.normalize(name)
            counts[normalized] += 1
            names.setdefault(normalized, name)

        for normalized, count in counts.items():
            existing = entities.get(normalized)
            if existing is not None:
                existing.mention_count += count
                # NER often mislabels titled names as ORG — the honorific wins.
                if existing.type != EntityType.PERSON:
                    existing.type = EntityType.PERSON
                continue
            name = names[normalized]
            entities[normalized] = Entity(
                id=Entity.make_id(name),
                name=name,
                normalized_name=normalized,
                type=EntityType.PERSON,
                mention_count=count,
            )

    # ------------------------------------------------------------------ #
    # Keyword extraction (lightweight, no heavy deps)
    # ------------------------------------------------------------------ #
    def _keywords(self, text: str, top_n: int = 12) -> list[str]:
        if not text:
            return []
        phrase_counts: Counter[str] = Counter()
        for match in _CAP_SEQ_RE.finditer(text):
            words = match.group(0).lower().split()
            while words and words[0] in _STOPWORDS:
                words.pop(0)
            while words and words[-1] in _STOPWORDS:
                words.pop()
            if not words:
                continue
            phrase = " ".join(words)
            if len(phrase) < 3 or phrase.isdigit():
                continue
            phrase_counts[phrase] += 1

        ranked = sorted(
            phrase_counts.items(),
            key=lambda kv: (-kv[1], -len(kv[0].split()), kv[0]),
        )
        keywords = [phrase for phrase, _ in ranked[:top_n]]

        if len(keywords) < top_n:
            chosen = set(keywords)
            unigrams: Counter[str] = Counter()
            for word in _WORD_RE.findall(text.lower()):
                if word in _STOPWORDS or word in chosen:
                    continue
                unigrams[word] += 1
            for word, _ in unigrams.most_common(top_n * 2):
                if word not in chosen:
                    keywords.append(word)
                    chosen.add(word)
                if len(keywords) >= top_n:
                    break

        return keywords[:top_n]

    # ------------------------------------------------------------------ #
    # Relation extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_predicate(sentence_lower: str) -> RelationType | None:
        for key, triggers in RELATION_TRIGGERS.items():
            for phrase in triggers:
                if phrase in sentence_lower:
                    try:
                        return RelationType(key)
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _pick_spoke_at_object(
        subject: Entity, positions: list[tuple[int, Entity]]
    ) -> Entity | None:
        """Choose the venue entity for a spoke_at edge, or None to skip it."""
        subject_norm = subject.normalized_name
        excluded_types = {
            EntityType.PERSON,
            EntityType.CITY,
            EntityType.STATE,
            EntityType.COUNTRY,
        }
        candidates: list[Entity] = []
        for _, entity in positions:
            if entity.id == subject.id or entity.type in excluded_types:
                continue
            norm = entity.normalized_name
            if norm in _ROLE_WORDS:
                continue
            # NER sometimes emits spans like "Commerce & Industry Piyush Goyal";
            # an object overlapping the speaker's name is a mangle, not a venue.
            if subject_norm in norm or norm in subject_norm:
                continue
            candidates.append(entity)
        if not candidates:
            return None
        # Prefer a true event over generic organisations when both are present.
        for entity in candidates:
            if entity.type == EntityType.EVENT:
                return entity
        return candidates[0]

    def _relations(
        self, text: str, entity_list: list[Entity], source_url: str
    ) -> list[Relation]:
        if len(entity_list) < 2:
            return []

        matchers: list[tuple[Entity, re.Pattern[str]]] = []
        for entity in entity_list:
            pattern = _compile_terms([entity.name, *entity.aliases])
            if pattern is not None:
                matchers.append((entity, pattern))

        # Newlines are hard sentence boundaries here: headlines and "Also Read:"
        # boilerplate rarely end with punctuation, and letting them merge into
        # the next sentence pairs subjects with entities they never co-occurred
        # with grammatically.
        sentences = [
            sentence
            for line in text.split("\n")
            for sentence in split_sentences(line)
        ]

        relations: dict[tuple[str, RelationType, str], Relation] = {}
        for sentence in sentences:
            predicate = self._match_predicate(sentence.lower())
            if predicate is None:
                continue

            positions: list[tuple[int, Entity]] = []
            for entity, pattern in matchers:
                match = pattern.search(sentence)
                if match is not None:
                    positions.append((match.start(), entity))
            if len(positions) < 2:
                continue

            positions.sort(key=lambda pair: pair[0])

            if predicate == RelationType.SPOKE_AT:
                # Person-gated: subject must be a PERSON, and the object a
                # venue-like entity (event/org/institution) — never a place
                # ("speaking … in Dar es Salaam"), a role word ("as CFO"), or a
                # mangled NER span that contains the speaker's own name.
                subject = next(
                    (e for _, e in positions if e.type == EntityType.PERSON), None
                )
                if subject is None:
                    continue
                obj = self._pick_spoke_at_object(subject, positions)
                if obj is None:
                    continue
            else:
                subject = positions[0][1]
                obj = None
                for _, candidate in positions[1:]:
                    if candidate.id != subject.id:
                        obj = candidate
                        break
                if obj is None:
                    continue

            key = (subject.id, predicate, obj.id)
            if key in relations:
                continue
            relations[key] = Relation(
                subject_id=subject.id,
                subject_name=subject.name,
                predicate=predicate,
                object_id=obj.id,
                object_name=obj.name,
                source_url=source_url,
                evidence=sentence.strip(),
                confidence=0.6,
            )
        return list(relations.values())
