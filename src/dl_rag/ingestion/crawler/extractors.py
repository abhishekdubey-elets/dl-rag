"""Metadata + content extraction from crawled HTML.

The primary metadata source is structured data (schema.org JSON-LD Article /
NewsArticle, then OpenGraph/meta tags); visible DOM selectors are the fallback.
Main content is extracted with ``trafilatura`` (markdown) and falls back to
:func:`dl_rag.ingestion.crawler.markdown.html_to_markdown` on the best content
container. Everything is assembled into a :class:`SourceDocument`.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from html import unescape
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import trafilatura
from bs4 import BeautifulSoup, Tag
from dateutil import parser as date_parser

from dl_rag.constants import CATEGORY_SLUG_TO_CONTENT_TYPE
from dl_rag.ingestion.crawler.markdown import html_to_markdown
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType
from dl_rag.utils.text import clean_whitespace, slugify

logger = get_logger(__name__)

# Article-ish schema.org @type values we treat as the main entity.
_ARTICLE_TYPES = {
    "article",
    "newsarticle",
    "blogposting",
    "report",
    "techarticle",
    "scholarlyarticle",
}

# Query params dropped when canonicalising URLs.
_TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "dclid",
    "gclsrc",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "ref_url",
    "source",
    "spm",
    "yclid",
    "_hsenc",
    "_hsmi",
}

_MONTHS = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

_MONTH_YEAR_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[\s\-–—/,]+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)

_ISSUE_YEAR_RE = re.compile(
    r"(?:issue|magazine|edition|vol(?:ume)?)[^\d]{0,24}((?:19|20)\d{2})",
    re.IGNORECASE,
)

_MAGAZINEISH = {ContentType.MAGAZINE_ISSUE, ContentType.MAGAZINE_ARTICLE}


# --------------------------------------------------------------------------- #
# Content-type routing
# --------------------------------------------------------------------------- #
def _looks_like_post_slug(slug: str) -> bool:
    """Heuristic: a long, hyphen-rich terminal slug looks like an article."""
    if not slug:
        return False
    if slug.isdigit():
        return False
    return slug.count("-") >= 2 or len(slug) >= 24


def _has_issue_slug(segments: list[str]) -> bool:
    joined = " ".join(segments).replace("-", " ")
    return bool(_MONTH_YEAR_RE.search(joined))


def detect_content_type(
    url: str, categories: list[str], tags: list[str]
) -> ContentType:
    """Classify a document from URL path segments + category/tag slugs.

    Slugs are matched against :data:`CATEGORY_SLUG_TO_CONTENT_TYPE`; a magazine
    landing slug that is followed by an article slug is upgraded to
    ``MAGAZINE_ARTICLE``. Falls back to keyword heuristics, then ``OTHER``.
    """
    path = urlsplit(url).path.lower()
    segments = [seg for seg in path.split("/") if seg]
    last = segments[-1] if segments else ""

    ordered_slugs = (
        [slugify(c) for c in categories if c]
        + [slugify(t) for t in tags if t]
        + segments
    )

    for slug in ordered_slugs:
        mapped = CATEGORY_SLUG_TO_CONTENT_TYPE.get(slug)
        if not mapped:
            continue
        content_type = ContentType(mapped)
        # A "magazine" *section* slug that precedes an article slug is an article.
        if (
            content_type is ContentType.MAGAZINE_ISSUE
            and slug != last
            and _looks_like_post_slug(last)
        ):
            return ContentType.MAGAZINE_ARTICLE
        return content_type

    blob = " ".join(ordered_slugs)
    if "interview" in blob:
        return ContentType.INTERVIEW
    if "magazine" in blob or _has_issue_slug(segments):
        return ContentType.MAGAZINE_ARTICLE
    if "news" in blob:
        return ContentType.NEWS
    if "nep" in blob or "policy" in blob:
        return ContentType.POLICY
    return ContentType.OTHER


# --------------------------------------------------------------------------- #
# Magazine issue parsing
# --------------------------------------------------------------------------- #
def _canonical_month(token: str) -> str | None:
    return _MONTHS.get(token.strip().lower())


def parse_issue(
    url: str, html: str, soup: BeautifulSoup
) -> tuple[str | None, str | None, int | None]:
    """Best-effort ``(issue_name, issue_month, issue_year)`` for magazine pages."""
    candidates: list[str] = []
    if soup is not None:
        heading = soup.select_one("h1")
        if heading:
            candidates.append(heading.get_text(" ", strip=True))
        og_title = _meta_content(soup, prop="og:title")
        if og_title:
            candidates.append(og_title)
        if soup.title and soup.title.string:
            candidates.append(soup.title.string)
    candidates.append(unquote(urlsplit(url).path).replace("-", " "))

    blob = "  ".join(c for c in candidates if c)

    issue_name: str | None = None
    issue_month: str | None = None
    issue_year: int | None = None

    match = _MONTH_YEAR_RE.search(blob)
    if match:
        issue_month = _canonical_month(match.group(1))
        issue_year = int(match.group(2))
        if issue_month:
            issue_name = f"{issue_month} {issue_year}"
    else:
        year_match = _ISSUE_YEAR_RE.search(blob)
        if year_match:
            issue_year = int(year_match.group(1))

    return issue_name, issue_month, issue_year


# --------------------------------------------------------------------------- #
# Metadata helpers
# --------------------------------------------------------------------------- #
def _meta_content(
    soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None
) -> str | None:
    tag: Any = None
    if prop is not None:
        tag = soup.find("meta", attrs={"property": prop})
    elif name is not None:
        tag = soup.find("meta", attrs={"name": name})
    if isinstance(tag, Tag):
        content = tag.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _iter_jsonld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Return every JSON-LD object on the page, flattening ``@graph`` arrays."""
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        stack: list[Any] = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                graph = item.get("@graph")
                if isinstance(graph, list):
                    stack.extend(graph)
                objects.append(item)
    return objects


def _type_matches(node: dict[str, Any]) -> bool:
    node_type = node.get("@type")
    if isinstance(node_type, str):
        return node_type.lower() in _ARTICLE_TYPES
    if isinstance(node_type, list):
        return any(isinstance(t, str) and t.lower() in _ARTICLE_TYPES for t in node_type)
    return False


def _jsonld_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(value, list):
        for item in value:
            name = _jsonld_name(item)
            if name:
                return name
    return None


def _jsonld_image(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        url = value.get("url") or value.get("@id")
        if isinstance(url, str) and url.strip():
            return url.strip()
    if isinstance(value, list):
        for item in value:
            url = _jsonld_image(item)
            if url:
                return url
    return None


def _jsonld_keywords(value: Any) -> list[str]:
    if isinstance(value, str):
        return [k.strip() for k in value.split(",") if k.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _to_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = date_parser.parse(value)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date()


def _is_tracking_param(key: str) -> bool:
    key = key.lower()
    return key.startswith("utm_") or key in _TRACKING_PARAMS


def _canonicalize_url(url: str, soup: BeautifulSoup | None = None) -> str:
    """Prefer ``rel=canonical`` if present, then strip tracking params/fragment."""
    target = url
    if soup is not None:
        link = soup.find("link", attrs={"rel": "canonical"})
        if isinstance(link, Tag):
            href = link.get("href")
            if isinstance(href, str) and href.strip():
                target = href.strip()

    parts = urlsplit(target)
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking_param(k)
    ]
    query = urlencode(kept)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get_text(" ", strip=True)
            if text:
                return text
    return None


def _collect_texts(soup: BeautifulSoup, selector: str) -> list[str]:
    out: list[str] = []
    for element in soup.select(selector):
        text = element.get_text(" ", strip=True)
        if text:
            out.append(text)
    return out


# --------------------------------------------------------------------------- #
# Main extractor
# --------------------------------------------------------------------------- #
def _extract_content_markdown(html: str, soup: BeautifulSoup) -> str:
    """Prefer trafilatura markdown; fall back to the best DOM container."""
    try:
        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=False,
            include_comments=False,
            favor_recall=True,
        )
    except Exception as exc:  # noqa: BLE001 - trafilatura can raise on odd input
        logger.debug("extract.trafilatura_failed", error=str(exc))
        extracted = None

    if extracted and extracted.strip():
        return clean_whitespace(extracted)

    container = (
        soup.select_one("article")
        or soup.select_one(".entry-content")
        or soup.select_one(".td-post-content")
        or soup.select_one("main")
        or soup.body
        or soup
    )
    return html_to_markdown(str(container))


def extract_document(
    url: str, html: str, crawled_at: datetime | None = None
) -> SourceDocument | None:
    """Extract a :class:`SourceDocument` from a page's URL + HTML.

    Returns ``None`` when the page yields neither a meaningful title nor content.
    ``crawled_at`` is passed through unchanged (the integrator/crawler sets it).
    """
    if not html or not html.strip():
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:  # noqa: BLE001 - malformed markup
        logger.warning("extract.parse_failed", url=url, error=str(exc))
        return None

    # --- structured data (primary) ---
    article_node: dict[str, Any] = {}
    for node in _iter_jsonld(soup):
        if _type_matches(node):
            article_node = node
            break

    jsonld_title = _jsonld_name(article_node.get("headline")) or (
        article_node.get("headline") if isinstance(article_node.get("headline"), str) else None
    )
    jsonld_published = article_node.get("datePublished")
    jsonld_modified = article_node.get("dateModified")
    jsonld_author = _jsonld_name(article_node.get("author"))
    jsonld_section = _jsonld_name(article_node.get("articleSection")) or (
        article_node.get("articleSection")
        if isinstance(article_node.get("articleSection"), str)
        else None
    )
    jsonld_image = _jsonld_image(article_node.get("image"))
    jsonld_keywords = _jsonld_keywords(article_node.get("keywords"))

    # --- OpenGraph / meta (secondary) ---
    og_title = _meta_content(soup, prop="og:title")
    og_description = _meta_content(soup, prop="og:description")
    og_image = _meta_content(soup, prop="og:image")
    og_type = _meta_content(soup, prop="og:type")
    meta_published = _meta_content(soup, prop="article:published_time")
    meta_modified = _meta_content(soup, prop="article:modified_time")
    meta_section = _meta_content(soup, prop="article:section")
    meta_keywords = _meta_content(soup, name="keywords")
    meta_author = _meta_content(soup, name="author")
    meta_description = _meta_content(soup, name="description")

    article_tags: list[str] = []
    for tag in soup.find_all("meta", attrs={"property": "article:tag"}):
        if isinstance(tag, Tag):
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                article_tags.append(content.strip())

    # --- visible DOM (fallback) ---
    dom_title = _first_text(soup, ["h1.entry-title", "h1.post-title", "h1", "title"])
    dom_author = _first_text(
        soup,
        [
            "a[rel=author]",
            ".entry-meta .author",
            ".author-name",
            ".td-post-author-name a",
            ".byline .author",
        ],
    )
    dom_date = None
    time_el = soup.select_one("time[datetime]")
    if isinstance(time_el, Tag) and isinstance(time_el.get("datetime"), str):
        dom_date = str(time_el.get("datetime"))
    if dom_date is None:
        dom_date = _first_text(soup, [".post-date", ".entry-date", ".td-post-date time"])

    dom_categories = _collect_texts(soup, ".cat-links a") or _collect_texts(
        soup, ".td-post-category"
    )
    dom_tags = _collect_texts(soup, ".tags-links a") or _collect_texts(soup, ".post-tags a")

    # --- resolve fields (structured → meta → DOM) ---
    title = jsonld_title or og_title or dom_title or ""
    title = clean_whitespace(unescape(title)) if title else ""

    subtitle_raw = og_description or meta_description
    subtitle = clean_whitespace(unescape(subtitle_raw)) if subtitle_raw else None

    author = jsonld_author or meta_author or dom_author
    author = clean_whitespace(unescape(author)) if author else None

    published_date = _to_date(jsonld_published) or _to_date(meta_published) or _to_date(dom_date)
    updated_date = _to_date(jsonld_modified) or _to_date(meta_modified)

    categories = [c for c in [jsonld_section, meta_section, *dom_categories] if c]
    tags = list(dict.fromkeys([*article_tags, *dom_tags]))

    featured_image = jsonld_image or og_image

    keywords: list[str] = list(jsonld_keywords)
    if meta_keywords:
        keywords.extend(k.strip() for k in meta_keywords.split(",") if k.strip())
    keywords = list(dict.fromkeys(keywords))

    canonical_url = _canonicalize_url(url, soup)
    content_markdown = _extract_content_markdown(html, soup)

    if not title.strip() and not content_markdown.strip():
        logger.debug("extract.empty", url=url)
        return None

    content_type = detect_content_type(canonical_url, categories, tags)

    issue_name = issue_month = None
    issue_year: int | None = None
    if content_type in _MAGAZINEISH or _has_issue_slug(
        [s for s in urlsplit(canonical_url).path.split("/") if s]
    ):
        issue_name, issue_month, issue_year = parse_issue(canonical_url, html, soup)

    metadata: dict[str, Any] = {
        "source": "dom",
        "canonical_url": canonical_url,
        "requested_url": url,
    }
    if og_type:
        metadata["og_type"] = og_type
    if article_node.get("@type"):
        metadata["schema_type"] = article_node["@type"]

    document = SourceDocument(
        id=SourceDocument.id_for_url(canonical_url),
        url=canonical_url,
        title=title or (subtitle or canonical_url),
        subtitle=subtitle,
        author=author,
        published_date=published_date,
        updated_date=updated_date,
        category=categories[0] if categories else None,
        content_type=content_type,
        tags=tags,
        content_markdown=content_markdown,
        featured_image=featured_image,
        issue_name=issue_name,
        issue_month=issue_month,
        issue_year=issue_year,
        keywords=keywords,
        metadata=metadata,
        crawled_at=crawled_at,
    )
    document.content_hash = document.compute_hash()
    return document
