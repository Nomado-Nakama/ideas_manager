from __future__ import annotations

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from nn_ideas_manager.core import _retriever, _vectorstore


# Small helper for tests
def num_vectors() -> int:
    return _vectorstore._collection.count()  # type: ignore[attr-defined]


# ------------------------------------------------------------------ #
#   High–level QA pipeline                                           #
# ------------------------------------------------------------------ #
_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant.
Use ONLY the context below to answer the question. If the answer is not
contained in the context, say you don't know.

--------------------
CONTEXT:
{context}
--------------------

QUESTION: {question}
ANSWER (markdown):"""
)

_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)  # cheap & fast


async def answer(question: str) -> BaseMessage:
    docs: list[Document] = _retriever.invoke(question)
    if not docs:
        return BaseMessage("🤷‍♂️ I don’t have any information about that yet.")

    context = "\n\n".join(d.page_content for d in docs)
    return _llm.invoke(_PROMPT.format(question=question, context=context))
