"""
Allow moderators to send a plain-text file with many Instagram links.
Each URL found in the file is saved to the `user_urls` table with the
same helper used by the single-link handler.
"""
from __future__ import annotations

import io
import re

from aiogram import Router, F
from aiogram.types import Message
from loguru import logger

from nn_ideas_manager.core.db.functions import save_url, get_telegram_user_role

router = Router()

# loose but serviceable – reuse the same pattern as the single-URL handler
_URL_RE = re.compile(r"https?://\S+")


@router.message(F.document.mime_type == "text/plain")
async def handle_bulk_upload(msg: Message) -> None:
    """Process a *.txt document sent by a moderator."""
    tg_id = msg.from_user.id
    role = await get_telegram_user_role(tg_id)
    if role != "moderator":
        await msg.answer("⛔ You do not have permission to ingest bulk links.")
        return

    # 1️⃣ download file into memory
    telegram_file = await msg.bot.get_file(msg.document.file_id)
    stream: io.BytesIO = io.BytesIO()
    await msg.bot.download_file(telegram_file.file_path, destination=stream)
    stream.seek(0)
    text = stream.read().decode("utf-8", errors="ignore")

    # 2️⃣ extract unique URLs
    urls: set[str] = {u.strip() for u in _URL_RE.findall(text)}
    if not urls:
        await msg.answer("⚠️ No links detected in the uploaded file.")
        return

    # 3️⃣ store URLs
    for url in urls:
        try:
            await save_url(tg_id, url)
        except Exception as exc:  # unlikely, but log duplicates/DB errors
            logger.warning("Could not save URL %s: %s", url, exc)

    await msg.answer(f"✅ Received {len(urls)} links — they’re now in the ingestion queue.")
