"""Enumerations shared across the codebase."""

from __future__ import annotations

from enum import Enum


class ContentType(str, Enum):
    """The kind of digitalLEARNING content a document represents."""

    MAGAZINE_ISSUE = "magazine_issue"
    MAGAZINE_ARTICLE = "magazine_article"
    NEWS = "news"
    INTERVIEW = "interview"
    RANKING = "ranking"
    POLICY = "policy"
    HIGHER_EDUCATION = "higher_education"
    SCHOOL = "school"
    GOVERNMENT_NEWS = "government_news"
    CORPORATE = "corporate"
    FEATURE = "feature"
    OPINION = "opinion"
    SPECIAL_STORY = "special_story"
    COVER_STORY = "cover_story"
    AUTHOR_PAGE = "author_page"
    TAG_PAGE = "tag_page"
    CATEGORY_PAGE = "category_page"
    VIDEO = "video"
    OTHER = "other"


class QueryType(str, Enum):
    """Detected user-query intent. Drives retrieval + prompt strategy."""

    TIMELINE = "timeline"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    TREND = "trend"
    POLICY = "policy"
    INSTITUTION = "institution"
    PERSON = "person"
    INTERVIEW = "interview"
    MAGAZINE = "magazine"
    EVENT = "event"
    RANKING = "ranking"
    RECOMMENDATION = "recommendation"
    STATISTICS = "statistics"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class EntityType(str, Enum):
    """Knowledge-graph entity categories."""

    UNIVERSITY = "university"
    SCHOOL = "school"
    GOVERNMENT_DEPARTMENT = "government_department"
    REGULATOR = "regulator"
    POLICY = "policy"
    SCHEME = "scheme"
    PERSON = "person"
    MINISTER = "minister"
    VICE_CHANCELLOR = "vice_chancellor"
    COMPANY = "company"
    EDTECH_STARTUP = "edtech_startup"
    PRODUCT = "product"
    STATE = "state"
    CITY = "city"
    COUNTRY = "country"
    ORGANIZATION = "organization"
    EVENT = "event"
    OTHER = "other"


class RelationType(str, Enum):
    """Canonical predicates in the knowledge graph."""

    INTRODUCED = "introduced"
    PARTNERED_WITH = "partnered_with"
    LAUNCHED = "launched"
    IMPLEMENTED = "implemented"
    LEADS = "leads"
    SPOKE_AT = "spoke_at"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    AFFILIATED_WITH = "affiliated_with"
    MENTIONED_WITH = "mentioned_with"
    RELATED_TO = "related_to"


class RetrievalSource(str, Enum):
    """Which retriever surfaced a candidate chunk."""

    DENSE = "dense"
    SPARSE = "sparse"
    KG = "kg"


class FeedbackRating(str, Enum):
    UP = "up"
    DOWN = "down"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
