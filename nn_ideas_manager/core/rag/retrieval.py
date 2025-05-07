from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate

from nn_ideas_manager.core.rag.multirepretriever import MultiRepRetriever
from nn_ideas_manager.core.rag import vector_store, doc_store, embeddings, llm


@dataclass(slots=True)
class AnswerResult:
    """Returned by `answer()` and consumed by the Telegram layer."""
    content: str
    context: str
    sources: list[str]


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
ANSWER:"""
)


async def answer(question: str, k: int = 3) -> AnswerResult:
    """
    Retrieves relevant docs, queries the LLM and returns the answer together
    with citations and the raw context.
    """

    retriever = MultiRepRetriever(vector_store, doc_store, embeddings)
    docs = await retriever.aretrieve(question, k)

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

    llm_reply: BaseMessage = llm.invoke(
        _PROMPT.format(question=question, context=context)
    )

    # Append foot-note style citations
    footnotes = ""
    if sources:
        footnote_lines = [f"{i + 1}. {url}" for i, url in enumerate(sources)]
        footnotes = "\n\n---\n**Sources**:\n" + "\n".join(footnote_lines)

    return AnswerResult(
        content=f"{llm_reply.content}{footnotes}",
        context=context,
        sources=sources,
    )
