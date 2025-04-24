"""
Worker that polls Postgres every 5 s for ``undigested`` links,
ingests them with LangChain pipeline, and updates status.

Run locally:
    python -m nn_ideas_manager.core.workers.process_undigested_urls
"""
from __future__ import annotations

import asyncio
from typing import Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from nn_ideas_manager.core.db.functions import (
    fetch_user_urls_by_status,
    mark_url_status,
)
from nn_ideas_manager.core.workers.process_undigested_urls.utils import ingestion_workflow, IngestResult


async def _handle_url(link_id: str, url: str, telegram_id: int) -> None:
    try:
        result: IngestResult = await ingestion_workflow(url)
        logger.info(
            "✅  Ingested {url} for user={u} → {n} chunks",
            url=url,
            u=telegram_id,
            n=len(result.doc_ids),
        )
        status: Literal["ingested"] = "ingested"
    except Exception as exc:
        logger.exception("💥  Failed to ingest {url}: {e}", url=url, e=exc)
        status: Literal["error"] = "error"

    await mark_url_status(link_id, status)


async def poll_undigested_urls() -> None:
    rows = await fetch_user_urls_by_status("undigested")
    if not rows:
        return

    # run concurrently but bounded
    await asyncio.gather(
        *(_handle_url(row["id"], row["url"], row["telegram_id"]) for row in rows)
    )


async def main() -> None:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_undigested_urls, trigger="interval", seconds=5)
    scheduler.start()
    logger.info("👷‍♂️  URL-ingestion worker started.")
    # Keep the event-loop alive forever
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
