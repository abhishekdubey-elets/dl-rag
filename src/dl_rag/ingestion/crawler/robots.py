"""robots.txt fetching and evaluation for polite crawling.

Wraps :class:`urllib.robotparser.RobotFileParser` with an async fetch step and a
fail-open policy: any error fetching or parsing ``/robots.txt`` is treated as
"allow all", and evaluation is a no-op when the checker is disabled or never
loaded. This keeps the crawler robust against missing/broken robots files while
still honouring a well-formed one.
"""

from __future__ import annotations

from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import httpx

from dl_rag.logging_config import get_logger

logger = get_logger(__name__)


class RobotsChecker:
    """Fetch and evaluate a site's ``robots.txt`` for a fixed user agent."""

    def __init__(self, base_url: str, user_agent: str, enabled: bool = True) -> None:
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent or "*"
        self._enabled = enabled
        self._parser: RobotFileParser | None = None
        self._loaded = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def load(self, client: httpx.AsyncClient) -> None:
        """Fetch and parse ``/robots.txt``; on any failure, treat as allow-all."""
        if not self._enabled:
            self._loaded = True
            return

        robots_url = urljoin(self._base_url + "/", "robots.txt")
        try:
            response = await client.get(
                robots_url,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - network failure → allow all
            logger.warning("robots.fetch_failed", url=robots_url, error=str(exc))
            self._parser = None
            self._loaded = True
            return

        if response.status_code >= 400:
            # No robots.txt (404) or server error → allow all.
            logger.info(
                "robots.absent", url=robots_url, status_code=response.status_code
            )
            self._parser = None
            self._loaded = True
            return

        parser = RobotFileParser()
        try:
            parser.parse(response.text.splitlines())
            self._parser = parser
        except Exception as exc:  # noqa: BLE001 - malformed robots → allow all
            logger.warning("robots.parse_failed", url=robots_url, error=str(exc))
            self._parser = None
        self._loaded = True

    def allowed(self, url: str) -> bool:
        """Return whether ``url`` may be fetched (always True if disabled/unloaded)."""
        if not self._enabled or not self._loaded or self._parser is None:
            return True
        try:
            return self._parser.can_fetch(self._user_agent, url)
        except Exception:  # noqa: BLE001 - never let robots eval crash a crawl
            return True
