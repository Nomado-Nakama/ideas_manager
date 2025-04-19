from aiogram import Router, F
from aiogram.types import Message
from nn_ideas_manager.core.tgbot.db.functions import save_user_link
import re

router = Router()
URL_RE = re.compile(r'https?://\S+')


@router.message(F.text.regexp(URL_RE.pattern))
async def link_handler(msg: Message):
    url = URL_RE.search(msg.text).group(0)
    await save_user_link(msg.from_user.id, url)
    await msg.answer("✅ Link stored!")
