"""`dl-ingest` — run the full crawl → chunk → embed → index pipeline.

Examples:
    poetry run dl-ingest --max-pages 200
    poetry run dl-ingest --content-type interview --content-type policy
    poetry run dl-ingest --url https://digitallearning.eletsonline.com/2022/06/...
    poetry run dl-ingest --full
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.logging_config import configure_logging, get_logger
from dl_rag.models.enums import ContentType

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crawl + ingest the digitalLEARNING archive.")
    p.add_argument("--url", dest="urls", action="append", default=[],
                   help="Explicit URL(s) to ingest (repeatable). Skips discovery.")
    p.add_argument("--content-type", dest="content_types", action="append", default=[],
                   choices=[c.value for c in ContentType],
                   help="Restrict to these content types (repeatable).")
    p.add_argument("--since", type=date.fromisoformat, default=None,
                   help="Only content published on/after YYYY-MM-DD.")
    p.add_argument("--max-pages", type=int, default=None, help="Cap pages crawled.")
    p.add_argument("--full", action="store_true", help="Full archive crawl.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)

    try:
        await container.db.create_all()
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest.schema_init_failed", error=str(exc))
        raise

    content_types = [ContentType(c) for c in args.content_types] or None
    try:
        stats = await container.pipeline.run(
            urls=args.urls or None,
            content_types=content_types,
            since_date=args.since,
            max_pages=args.max_pages,
            full_crawl=args.full,
        )
        print("\nIngestion complete:")
        for key in sorted(stats):
            print(f"  {key:16s}: {stats[key]}")
    finally:
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
