import os
import loguru
import asyncio

from nn_ideas_manager.core.tgbot.db.functions import (
    fetch_user_links_by_status,
    mark_link_status,
)

logger = loguru.logger


async def poll_undigested_urls():
    """
    Every 30s: grab all 'unprocessed' links, process them,
    then mark 'processed' or 'error'.
    """
    undigested_urls = await fetch_user_links_by_status("undigested")
    if not undigested_urls:
        return

    for undigested_url in undigested_urls:
        link_id = undigested_url["id"]
        url = undigested_url["url"]
        tg_id = undigested_url["telegram_id"]

        try:
            logger.info(f"Processing {url} for user {tg_id}")
            # ── your actual processing logic here ──
            # 1. Download content by url
            # 2. Extract metadata if available
            # 3. Extract video -> audio -> text
            # 4. Extract video -> images -> text
            # 5. Extract image -> text
            # 6. Combine text from 2 - 5 and embedd
            # 7. Store embeddings in ChromaDB
            # ── mark as done ──
            await mark_link_status(link_id, "ingested")

        except Exception as e:
            logger.error(f"Failed to process link {link_id}: {e}")
            await mark_link_status(link_id, "error")
