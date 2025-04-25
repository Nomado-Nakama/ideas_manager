import asyncio
from aiogram import Bot, Dispatcher
from nn_ideas_manager.core.tgbot.config import BOT_TOKEN
from nn_ideas_manager.core.tgbot.handlers.start_handler import router as cmd_router
from nn_ideas_manager.core.tgbot.handlers.url_handler import router as url_router
from nn_ideas_manager.core.tgbot.handlers.ask_handler import router as ask_router


async def main():
    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(cmd_router)
    dp.include_router(url_router)
    dp.include_router(ask_router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
