from __future__ import annotations

from dataclasses import dataclass

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from nn_ideas_manager.core import _retriever, _vectorstore


@dataclass(slots=True)
class AnswerResult:
    """Returned by `answer()` and consumed by the Telegram layer."""
    content: str          # LLM answer (markdown) with citations appended
    context: str          # Exact text that was provided to the LLM
    sources: list[str]    # Unique list of `source_url` values


# ------------------------------------------------------------------ #
#   Helper for tests                                                 #
# ------------------------------------------------------------------ #
def num_vectors() -> int:
    # type: ignore[attr-defined]
    return _vectorstore._collection.count()


# ------------------------------------------------------------------ #
#   High-level QA pipeline                                           #
# ------------------------------------------------------------------ #
_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant.
Use ONLY the context below to answer the question.

--------------------
CONTEXT:
{context}
--------------------

QUESTION: {question}
ANSWER (markdown):"""
)

_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)  # cheap & fast


async def answer(question: str) -> AnswerResult:
    """
    Retrieves relevant docs, queries the LLM and returns the answer together
    with citations and the raw context.
    """
    docs: list[Document] = _retriever.invoke(question)

    if not docs:
        return AnswerResult(
            content="🤷‍♂️ I don’t have any information about that yet.",
            context="",
            sources=[],
        )

    # Build context and collect unique `source_url`s
    context = "\n\n".join(d.page_content for d in docs)
    sources: list[str] = []
    for d in docs:
        src = d.metadata.get("source_url")
        if src and src not in sources:
            sources.append(src)

    llm_reply: BaseMessage = _llm.invoke(
        _PROMPT.format(question=question, context=context)
    )

    # Append foot-note style citations
    footnotes = ""
    if sources:
        footnote_lines = [f"[{i + 1}]. {url}" for i, url in enumerate(sources)]
        footnotes = "\n\n---\n**Sources**:\n" + "\n".join(footnote_lines)

    return AnswerResult(
        content=f"{llm_reply.content}{footnotes}",
        context=context,
        sources=sources,
    )
