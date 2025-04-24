import asyncio

import loguru
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from nn_ideas_manager.core.db.functions import (
    fetch_user_urls_by_status,
    mark_url_status,
)

logger = loguru.logger


async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_undigested_urls, trigger="interval", seconds=5)
    scheduler.start()


async def ingestion_workflow(url):
    """
    1. Download content by url
    2. Extract metadata if available
    3. Extract video -> audio -> text
    4. Extract video -> images -> text
    5. Extract image -> text
    6. Combine text from 2 - 5 and embedd
    7. Store embeddings in ChromaDB
    """
    raise NotImplemented()


async def poll_undigested_urls():
    """
    Every 5s: grab all 'undigested' links, process them,
    then mark 'processed' or 'error'.
    """
    undigested_urls = await fetch_user_urls_by_status("undigested")
    if not undigested_urls:
        return

    for undigested_url in undigested_urls:
        link_id = undigested_url["id"]
        url = undigested_url["url"]
        tg_id = undigested_url["telegram_id"]

        try:
            logger.info(f"Processing {url} for user {tg_id}")
            new_status = await ingestion_workflow(url)
            await mark_url_status(link_id, )

        except Exception as e:
            logger.error(f"Failed to process link {link_id}: {e}")
            await mark_url_status(link_id, "error")


if __name__ == "__main__":
    asyncio.run(main())
