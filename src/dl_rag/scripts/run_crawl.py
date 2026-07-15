"""`dl-crawl` — discover archive URLs and optionally fetch a sample document.

Useful for validating the crawler's selectors against the live digitalLEARNING
site before committing to a full ingest. Does NOT write to any datastore.

Examples:
    poetry run dl-crawl --max-pages 100 --limit 20
    poetry run dl-crawl --limit 5 --fetch
"""

from __future__ import annotations

import argparse
import asyncio

from dl_rag.config import get_settings
from dl_rag.ingestion.crawler.wordpress import WordPressCrawler
from dl_rag.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover (and optionally sample) archive URLs.")
    p.add_argument("--max-pages", type=int, default=None, help="Cap discovered URLs.")
    p.add_argument("--limit", type=int, default=25, help="How many URLs to print.")
    p.add_argument("--fetch", action="store_true", help="Fetch + parse the first URL as a sample.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    crawler = WordPressCrawler(settings)

    urls = await crawler.discover_urls(max_pages=args.max_pages)
    print(f"Discovered {len(urls)} URLs from {settings.crawler_base_url}\n")
    for url in urls[: args.limit]:
        print(" ", url)

    if args.fetch and urls:
        print("\nFetching sample:", urls[0])
        async for doc in crawler.crawl([urls[0]]):
            print(f"  title       : {doc.title}")
            print(f"  content_type: {doc.content_type.value}")
            print(f"  author      : {doc.author}")
            print(f"  published   : {doc.published_date}")
            print(f"  words       : {doc.word_count}")
            print(f"  tags        : {doc.tags}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
