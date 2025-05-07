from langchain.schema import Document


class MultiRepRetriever:
    def __init__(self, vs, doc_store, embeddings):
        self._vs = vs
        self._doc_store = doc_store
        self._embeddings = embeddings

    async def aretrieve(self, query: str, k: int = 5) -> list[Document]:
        # 1️⃣ embed the query
        q_emb = self._embeddings.embed_query(query)

        # 2️⃣ similarity search on summaries
        hits = self._vs.similarity_search_by_vector(q_emb, k=k)

        # 3️⃣ collect doc_ids and original metadata
        doc_ids = [h.metadata["doc_id"] for h in hits]
        urls = [h.metadata["source_url"] for h in hits]

        # 4️⃣ fetch raw docs
        full_docs = await self._doc_store.mget(doc_ids)

        # 5️⃣ wrap for downstream RAG
        return [
            Document(page_content=txt,
                     metadata={"doc_id": d, "source_url": u})
            for txt, d, u in zip(full_docs, doc_ids, urls)
        ]
