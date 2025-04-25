from aiogram import Router
from aiogram.filters import CommandObject, Command
from aiogram.types import Message
from nn_ideas_manager.core.retrieval import answer

router = Router()


@router.message(Command("ask"))
async def ask_handler(msg: Message, command: CommandObject) -> None:
    query = (command.args or "").strip()
    if not query:
        await msg.answer("ℹ️  Usage: `/ask Ask any question to the nakama knowledge base`")
        return

    await msg.answer("🔎 Let me think…")
    reply = await answer(query)
    await msg.answer(f"Meta data: {str(reply.response_metadata)}")
    await msg.answer(reply.content)
