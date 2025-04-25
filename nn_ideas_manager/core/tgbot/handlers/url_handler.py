from aiogram import Router, F
from aiogram.types import Message
import re
from nn_ideas_manager.core.db.functions import (
    save_url,
    get_telegram_user_role
)

router = Router()
URL_RE = re.compile(r'https?://\S+')


@router.message(F.text.regexp(URL_RE.pattern))
async def link_handler(msg: Message):
    tg_id = msg.from_user.id
    role = await get_telegram_user_role(tg_id)
    if role != "moderator":
        await msg.answer("⛔ You do not have permission to ingest links.")
        return

    url = URL_RE.search(msg.text).group(0)
    await save_url(tg_id, url)
    await msg.answer("✅ Link sent for ingestion!")
