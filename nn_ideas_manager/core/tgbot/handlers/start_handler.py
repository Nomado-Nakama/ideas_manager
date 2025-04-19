from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from nn_ideas_manager.core.tgbot.db.functions import save_telegram_user

router = Router()


@router.message(Command("start"))
async def cmd_start_handler(msg: Message):
    # 1. Save the user in our DB
    tg_user = msg.from_user.model_dump()
    # map to our column names:
    user_dict = {
        "id": tg_user["id"],
        "first_name": tg_user.get("first_name"),
        "last_name": tg_user.get("last_name"),
    }
    await save_telegram_user(user_dict)

    # 2. Greet them
    await msg.answer(
        "👋 Hello! Send me an Instagram link and I'll store it for you."
    )
