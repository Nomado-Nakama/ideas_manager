import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv(r'\Projects\python\nakama-ideas-manager\configs\.env')

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
assert _OPENAI_API_KEY, "OPENAI_API_KEY must be set in environment"

_CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", r"\Projects\python\nakama-ideas-manager\chroma")
).resolve()
_CHROMA_DIR.mkdir(exist_ok=True, parents=True)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_vectorstore = Chroma(
    persist_directory=str(_CHROMA_DIR),
    embedding_function=_embeddings,  # same embedding model
)

_retriever = _vectorstore.as_retriever(
    search_type="mmr",  # same policy as on the diagram
    search_kwargs={"k": 3, "lambda_mult": 0.8},
)
