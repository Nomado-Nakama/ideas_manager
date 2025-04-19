from aiogram import Router, F
from aiogram.types import Message
from nn_ideas_manager.core.tgbot.db.functions import save_user_link, get_telegram_user_role
import re

router = Router()
URL_RE = re.compile(r'https?://\S+')


@router.message(F.text.regexp(URL_RE.pattern))
async def link_handler(msg: Message):
    tg_id = msg.from_user.id

    # 1. Look up the user's role
    role = await get_telegram_user_role(tg_id)
    if role != "moderator":
        # 2a. Deny
        await msg.answer("⛔ You do not have permission to store links.")
        return

    # 2b. Allow
    url = URL_RE.search(msg.text).group(0)
    await save_user_link(tg_id, url)
    await msg.answer("✅ Link stored!")
