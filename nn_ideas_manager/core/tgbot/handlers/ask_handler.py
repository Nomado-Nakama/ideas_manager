from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandObject, Command
from aiogram.types import Message, BufferedInputFile
import io

from nn_ideas_manager.core.retrieval import answer, AnswerResult

router = Router()


@router.message(Command("ask"))
async def ask_handler(msg: Message, command: CommandObject) -> None:
    query = (command.args or "").strip()
    if not query:
        await msg.answer(
            "ℹ️ Usage: `/ask Ask any question to the nakama knowledge base`",
            parse_mode="Markdown",
        )
        return

    await msg.answer("🔎 Let me think…")

    result: AnswerResult = await answer(query)

    # 1️⃣ Send the raw context as a downloadable file
    if result.context:
        buf = io.BytesIO(result.context.encode("utf-8"))
        doc = BufferedInputFile(buf.getvalue(), filename="context.txt")
        await msg.answer_document(
            doc,
            caption="📄 *Full context used to answer your question*",
            parse_mode="Markdown",
        )

    # 2️⃣ Send the LLM answer (already contains citations)
    try:
        await msg.answer(result.content, parse_mode="Markdown")
    except Exception:
        await msg.answer(result.content)
