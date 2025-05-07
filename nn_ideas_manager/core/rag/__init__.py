import os
from pathlib import Path

from dotenv import load_dotenv

from chromadb import HttpClient
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from nn_ideas_manager.core.rag.docstore import SQLDocStore
from nn_ideas_manager.core.db.connection import get_pool

load_dotenv('/Projects/python/nakama-ideas-manager/configs/.env')

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
assert _OPENAI_API_KEY, "OPENAI_API_KEY must be set in environment"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", "/Projects/python/nakama-ideas-manager/chroma")
).resolve()
CHROMA_DIR.mkdir(exist_ok=True, parents=True)

CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", None)
assert CHROMA_DB_HOST, "CHROMA_DB_HOST must be set in environment"

CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT", None)
assert CHROMA_DB_HOST, "CHROMA_DB_PORT must be set in environment"

CHROMA_DB_SSL = os.getenv("CHROMA_DB_SSL", None)
assert CHROMA_DB_HOST, "CHROMA_DB_SSL must be set in environment"

CHROMA_DB_SSL = bool(CHROMA_DB_SSL) if CHROMA_DB_SSL.lower() == 'true' else False

vector_store = Chroma(
    persist_directory=str(CHROMA_DIR),
    client=HttpClient(
        host=CHROMA_DB_HOST, port=int(CHROMA_DB_PORT), ssl=CHROMA_DB_SSL
    ),
    embedding_function=embeddings,
)

doc_store = SQLDocStore(get_pool=get_pool)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
summarizer = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
