"""`dl-auto-ingest` — run the auto-ingest scheduler outside the API process.

Same logic the API runs in-process when ``AUTO_INGEST_ENABLED=true``; use this
form for a cron entry (``--once``) or as a dedicated worker container so
embedding work never competes with chat latency.

Examples:
    poetry run dl-auto-ingest --once          # one run, print the summary, exit
    poetry run dl-auto-ingest                 # loop forever on AUTO_INGEST_INTERVAL_HOURS
"""

from __future__ import annotations

import argparse
import asyncio
import json

from dl_rag.api.deps import build_container
from dl_rag.config import get_settings
from dl_rag.logging_config import configure_logging


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-ingest new articles and channel videos.")
    p.add_argument("--once", action="store_true", help="Run a single pass and exit.")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()
    container = build_container(settings)
    service = container.auto_ingest
    try:
        if args.once:
            result = await service.run_once(reason="cli")
            print(json.dumps(result, indent=2, default=str))
            return
        interval = max(60.0, settings.auto_ingest_interval_hours * 3600.0)
        while True:
            result = await service.run_once(reason="cli-loop")
            print(json.dumps(result, default=str))
            await asyncio.sleep(interval)
    finally:
        await container.db.dispose()
        await container.cache.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
