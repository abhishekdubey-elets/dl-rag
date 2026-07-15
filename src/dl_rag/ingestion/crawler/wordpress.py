"""Async WordPress crawler: URL discovery, REST-API mapping, and page fetching.

Discovery unions the WordPress REST API (``/wp-json/wp/v2/posts`` + ``/pages``,
``_embed``-ed, header-paginated) with XML sitemaps (Yoast ``sitemap_index.xml``
or a flat ``sitemap.xml``). Fetching is bounded-concurrency and polite, with
tenacity retries on transient failures. Everything is fully async and shares a
single :class:`httpx.AsyncClient` per operation.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from dl_rag.config import Settings
from dl_rag.ingestion.crawler.extractors import (
    _MAGAZINEISH,
    _canonicalize_url,
    detect_content_type,
    extract_document,
    parse_issue,
)
from dl_rag.ingestion.crawler.markdown import html_to_markdown
from dl_rag.ingestion.crawler.robots import RobotsChecker
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType

logger = get_logger(__name__)

_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE | re.DOTALL)
# Path prefixes that are listings/taxonomies, not individual articles.
_NON_ARTICLE_SEGMENTS = {
    "category",
    "tag",
    "author",
    "page",
    "wp-content",
    "wp-json",
    "feed",
    "comments",
    "search",
    # Utility / listing pages — no article content; they pollute retrieval
    # (endless title lists + crawl-date freshness) if ingested.
    "videos",
    "video",
    "video-gallery",
    "video-gallery-detail",
    "all-videos",
    "event-gallery",
    "photos",
    "photo-gallery",
    "magazinepdf",
    "advertise",
    "write-for-us",
    "contact-us",
    "subscribe",
    "sitemap",
}
_API_PAGE_SAFETY_CAP = 100


def _is_transient(exc: BaseException) -> bool:
    """Retry timeouts, transport errors, and 5xx responses only."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _html_to_text(html: str) -> str:
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "lxml")
    return " ".join(soup.get_text(" ", strip=True).split())


def _looks_like_article(url: str) -> bool:
    segments = [s for s in urlsplit(url).path.split("/") if s]
    if not segments:
        return False
    return not any(seg in _NON_ARTICLE_SEGMENTS for seg in segments)


class WordPressCrawler:
    """Discover and fetch article documents from a WordPress site."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.base_url = settings.crawler_base_url.rstrip("/")
        self.user_agent = settings.crawler_user_agent
        self.concurrency = max(1, settings.crawler_concurrency)
        self.delay = max(0.0, settings.crawler_delay_seconds)
        self.timeout = settings.crawler_timeout_seconds
        self.respect_robots = settings.crawler_respect_robots
        self.max_pages = settings.crawler_max_pages

    # ------------------------------------------------------------------ #
    # Client
    # ------------------------------------------------------------------ #
    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=httpx.Timeout(float(self.timeout)),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=self.concurrency * 2),
        )

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    async def discover_urls(
        self,
        *,
        content_types: list[ContentType] | None = None,
        since_date: date | None = None,
        max_pages: int | None = None,
    ) -> list[str]:
        """Union of REST-API and sitemap article URLs, deduped + robots-filtered."""
        resolved = max_pages if max_pages is not None else self.max_pages
        limit = resolved if resolved and resolved > 0 else None
        if content_types:
            logger.info(
                "crawler.discover.content_type_hint",
                types=[c.value for c in content_types],
            )

        ordered: dict[str, None] = {}
        async with self._new_client() as client:
            robots = RobotsChecker(self.base_url, self.user_agent, self.respect_robots)
            try:
                await robots.load(client)
            except Exception as exc:  # noqa: BLE001 - fail open
                logger.warning("crawler.robots_load_failed", error=str(exc))

            try:
                for link in await self._api_discover(client, since_date, limit):
                    ordered.setdefault(link, None)
            except Exception as exc:  # noqa: BLE001 - one source failing is OK
                logger.warning("crawler.api_discover_failed", error=str(exc))

            try:
                for link in await self._sitemap_discover(client):
                    ordered.setdefault(link, None)
            except Exception as exc:  # noqa: BLE001 - one source failing is OK
                logger.warning("crawler.sitemap_discover_failed", error=str(exc))

            urls = [u for u in ordered if robots.allowed(u)]

        if limit is not None:
            urls = urls[:limit]
        logger.info("crawler.discover.done", discovered=len(urls))
        return urls

    async def _api_discover(
        self,
        client: httpx.AsyncClient,
        since_date: date | None,
        limit: int | None,
    ) -> list[str]:
        found: list[str] = []
        for endpoint in ("posts", "pages"):
            page = 1
            while True:
                params: dict[str, Any] = {"per_page": 100, "page": page, "_embed": 1}
                if since_date is not None:
                    params["after"] = f"{since_date.isoformat()}T00:00:00"
                api_url = f"{self.base_url}/wp-json/wp/v2/{endpoint}"
                try:
                    response = await client.get(api_url, params=params)
                except httpx.HTTPError as exc:
                    logger.warning(
                        "crawler.api_request_failed",
                        endpoint=endpoint,
                        page=page,
                        error=str(exc),
                    )
                    break
                # WP returns 400 ("rest_post_invalid_page_number") past the last page.
                if response.status_code >= 400:
                    break
                try:
                    data = response.json()
                except ValueError:
                    break
                if not isinstance(data, list) or not data:
                    break

                for post in data:
                    if not isinstance(post, dict):
                        continue
                    link = post.get("link")
                    if isinstance(link, str) and link:
                        found.append(link)
                        if limit is not None and len(found) >= limit:
                            return found

                total_pages = _int_header(response.headers.get("X-WP-TotalPages"))
                if total_pages and page >= total_pages:
                    break
                if page >= _API_PAGE_SAFETY_CAP:
                    break
                page += 1
        return found

    async def _sitemap_discover(self, client: httpx.AsyncClient) -> list[str]:
        found: list[str] = []
        index_url = urljoin(self.base_url + "/", "sitemap_index.xml")
        child_sitemaps: list[str] = []

        try:
            response = await client.get(index_url)
            if response.status_code < 400 and "<loc>" in response.text:
                for loc in _LOC_RE.findall(response.text):
                    loc = unescape(loc.strip())
                    if loc.endswith(".xml"):
                        child_sitemaps.append(loc)
        except httpx.HTTPError as exc:
            logger.info("crawler.sitemap_index_missing", error=str(exc))

        if child_sitemaps:
            # Prefer post/page sitemaps; skip taxonomy sitemaps entirely.
            for sitemap_url in child_sitemaps:
                low = sitemap_url.lower()
                if any(seg in low for seg in ("category", "tag", "author")):
                    continue
                found.extend(await self._read_sitemap(client, sitemap_url))
        else:
            flat_url = urljoin(self.base_url + "/", "sitemap.xml")
            found.extend(await self._read_sitemap(client, flat_url))

        return [u for u in dict.fromkeys(found) if _looks_like_article(u)]

    async def _read_sitemap(self, client: httpx.AsyncClient, url: str) -> list[str]:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            logger.info("crawler.sitemap_fetch_failed", url=url, error=str(exc))
            return []
        if response.status_code >= 400:
            return []
        return [unescape(loc.strip()) for loc in _LOC_RE.findall(response.text)]

    # ------------------------------------------------------------------ #
    # REST-API document mapping (richer path)
    # ------------------------------------------------------------------ #
    async def fetch_via_api(self, post: dict[str, Any]) -> SourceDocument | None:
        """Map an ``_embed``-ed WP REST post object straight to a SourceDocument."""
        try:
            link = post.get("link")
            if not isinstance(link, str) or not link:
                return None

            title = _rendered(post.get("title"))
            title = " ".join(unescape(title).split())
            subtitle = _html_to_text(_rendered(post.get("excerpt"))) or None
            content_html = _rendered(post.get("content"))

            published = _wp_date(post.get("date") or post.get("date_gmt"))
            updated = _wp_date(post.get("modified") or post.get("modified_gmt"))

            embedded = post.get("_embedded") or {}
            author = _embedded_author(embedded)
            category_names, category_slugs, tag_names, tag_slugs = _embedded_terms(embedded)
            featured_image = _embedded_media(embedded)

            content_markdown = (
                await asyncio.to_thread(html_to_markdown, content_html)
                if content_html
                else ""
            )

            canonical_url = _canonicalize_url(link)
            content_type = detect_content_type(
                canonical_url,
                category_slugs or category_names,
                tag_slugs or tag_names,
            )

            issue_name = issue_month = None
            issue_year: int | None = None
            if content_type in _MAGAZINEISH:
                soup = BeautifulSoup(content_html or "", "lxml")
                issue_name, issue_month, issue_year = parse_issue(
                    canonical_url, content_html, soup
                )

            if not title.strip() and not content_markdown.strip():
                return None

            document = SourceDocument(
                id=SourceDocument.id_for_url(canonical_url),
                url=canonical_url,
                title=title or (subtitle or canonical_url),
                subtitle=subtitle,
                author=author,
                published_date=published,
                updated_date=updated,
                category=category_names[0] if category_names else None,
                content_type=content_type,
                tags=tag_names,
                content_markdown=content_markdown,
                featured_image=featured_image,
                issue_name=issue_name,
                issue_month=issue_month,
                issue_year=issue_year,
                keywords=[],
                metadata={
                    "source": "wp-api",
                    "wp_id": post.get("id"),
                    "canonical_url": canonical_url,
                },
                crawled_at=datetime.now(tz=UTC),
            )
            document.content_hash = document.compute_hash()
            return document
        except Exception as exc:  # noqa: BLE001 - never let one post kill the batch
            logger.warning(
                "crawler.api_map_failed", link=post.get("link"), error=str(exc)
            )
            return None

    # ------------------------------------------------------------------ #
    # Page fetching (DOM path)
    # ------------------------------------------------------------------ #
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8.0),
        retry=retry_if_exception(_is_transient),
    )
    async def _get_html(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.get(
            url,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text

    async def fetch_document(
        self, client: httpx.AsyncClient, url: str
    ) -> SourceDocument | None:
        """GET ``url`` and extract a SourceDocument; retry transient failures."""
        try:
            html = await self._get_html(client, url)
        except Exception as exc:  # noqa: BLE001 - exhausted retries / 4xx
            logger.warning("crawler.fetch_failed", url=url, error=str(exc))
            return None

        crawled_at = datetime.now(tz=UTC)
        document = await asyncio.to_thread(extract_document, url, html, crawled_at)
        if document is None:
            logger.debug("crawler.no_document", url=url)
        return document

    # ------------------------------------------------------------------ #
    # Bounded-concurrency crawl
    # ------------------------------------------------------------------ #
    async def crawl(self, urls: Sequence[str]) -> AsyncIterator[SourceDocument]:
        """Fetch ``urls`` with bounded concurrency, yielding documents as ready."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async with self._new_client() as client:

            async def worker(target: str) -> SourceDocument | None:
                async with semaphore:
                    document = await self.fetch_document(client, target)
                    if self.delay:
                        await asyncio.sleep(self.delay)
                    return document

            tasks = [asyncio.create_task(worker(u)) for u in urls]
            try:
                for completed in asyncio.as_completed(tasks):
                    try:
                        document = await completed
                    except Exception as exc:  # noqa: BLE001 - skip + log failures
                        logger.warning("crawler.worker_failed", error=str(exc))
                        continue
                    if document is not None:
                        yield document
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()


# --------------------------------------------------------------------------- #
# Module-level helpers for WP REST shapes
# --------------------------------------------------------------------------- #
def _int_header(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rendered(field: Any) -> str:
    if isinstance(field, dict):
        rendered = field.get("rendered")
        if isinstance(rendered, str):
            return rendered
    if isinstance(field, str):
        return field
    return ""


def _wp_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _embedded_author(embedded: dict[str, Any]) -> str | None:
    authors = embedded.get("author")
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict):
            name = first.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _embedded_terms(
    embedded: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    category_names: list[str] = []
    category_slugs: list[str] = []
    tag_names: list[str] = []
    tag_slugs: list[str] = []
    term_groups = embedded.get("wp:term")
    if isinstance(term_groups, list):
        for group in term_groups:
            if not isinstance(group, list):
                continue
            for term in group:
                if not isinstance(term, dict):
                    continue
                taxonomy = term.get("taxonomy")
                name = term.get("name")
                slug = term.get("slug")
                if taxonomy == "category":
                    if isinstance(name, str) and name:
                        category_names.append(name)
                    if isinstance(slug, str) and slug:
                        category_slugs.append(slug)
                elif taxonomy == "post_tag":
                    if isinstance(name, str) and name:
                        tag_names.append(name)
                    if isinstance(slug, str) and slug:
                        tag_slugs.append(slug)
    return category_names, category_slugs, tag_names, tag_slugs


def _embedded_media(embedded: dict[str, Any]) -> str | None:
    media = embedded.get("wp:featuredmedia")
    if isinstance(media, list) and media:
        first = media[0]
        if isinstance(first, dict):
            source_url = first.get("source_url")
            if isinstance(source_url, str) and source_url.strip():
                return source_url.strip()
    return None
