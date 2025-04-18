from aiogram import Router, F
from aiogram.types import Message
import re
from nn_ideas_manager.core.tgbot.db.functions import (
    save_user_link,
    get_telegram_user_role
)

router = Router()
URL_RE = re.compile(r'https?://\S+')


@router.message(F.text.regexp(URL_RE.pattern))
async def link_handler(msg: Message):
    tg_id = msg.from_user.id
    role = await get_telegram_user_role(tg_id)
    if role != "moderator":
        await msg.answer("⛔ You do not have permission to store links.")
        return

    url = URL_RE.search(msg.text).group(0)
    await save_user_link(tg_id, url)
    await msg.answer("✅ Link stored!")
