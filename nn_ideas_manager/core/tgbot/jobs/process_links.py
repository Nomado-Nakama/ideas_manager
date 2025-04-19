import loguru

from nn_ideas_manager.core.tgbot.db.functions import (
    fetch_user_links_by_status,
    mark_link_status,
)

from pathlib import Path

from knowledge_devourer.core.config import Config
from knowledge_devourer.core.processor import process_posts, process_reels

logger = loguru.logger


async def poll_unprocessed_links():
    """
    Every 30s: grab all 'unprocessed' links, process them,
    then mark 'processed' or 'error'.
    """
    rows = await fetch_user_links_by_status("unprocessed")
    if not rows:
        return

    for row in rows:
        link_id = row["id"]
        url = row["url"]
        tg_id = row["telegram_id"]

        try:
            # ── your actual processing logic here ──
            logger.info(f"Processing {url} for user {tg_id}")

            await use_kd(url=url)

            # ── mark as done ──
            await mark_link_status(link_id, "processed")

        except Exception as e:
            logger.error(f"Failed to process link {link_id}: {e}")
            await mark_link_status(link_id, "error")


async def use_kd(url: str):
    config = Config(
        base_dir=Path("/app/storage")
    )

    process_posts(
        links=[url],
        config=config
    )  # handle /p/ links

    process_reels(
        links=[url],
        config=config
    )  # handle /reel/ links


