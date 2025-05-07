from aiogram import Router
from aiogram.filters import CommandObject, Command
from aiogram.types import Message, BufferedInputFile
import io

from nn_ideas_manager.core.rag.retrieval import answer, AnswerResult

router = Router()


@router.message(Command("ask"))
async def ask_handler(msg: Message, command: CommandObject) -> None:
    """
    `/ask [num_sources] question …`

    * `/ask Where was this filmed?`      – defaults to 3 sources
    * `/ask 5 Where was this filmed?`    – tries to retrieve 5 sources
    """
    raw = (command.args or "").strip()

    if not raw:
        await msg.answer(
            "ℹ️ Usage:\n"
            "• `/ask Where was this filmed?`\n"
            "• `/ask 5 Where was this filmed?`",
            parse_mode="Markdown",
        )
        return

    first, *rest = raw.split(maxsplit=1)

    # try to interpret the first token as an int
    try:
        k = max(1, min(10, int(first)))   # keep it in a sensible range (1-10)
        query = rest[0] if rest else ""
    except ValueError:
        k = 5                              # fallback to default
        query = raw

    if not query:
        await msg.answer("❗ I need a question after the number.")
        return

    await msg.answer(f"🔎 Looking for up to *{k}* relevant chunks…", parse_mode="Markdown")

    result: AnswerResult = await answer(query, k=k)

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
