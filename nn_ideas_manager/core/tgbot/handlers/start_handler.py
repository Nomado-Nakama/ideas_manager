from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from nn_ideas_manager.core.db.functions import save_telegram_user

router = Router()


@router.message(Command("start"))
async def cmd_start_handler(msg: Message):
    user = {
        "id": msg.from_user.id,
        "first_name": msg.from_user.first_name,
        "last_name": msg.from_user.last_name
    }
    await save_telegram_user(user)
    await msg.answer("👋 Hello! Send me an Instagram link and I’ll ingest it so you could ask me about it's contents.")
