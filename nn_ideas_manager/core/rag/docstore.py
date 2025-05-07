from typing import Callable


class SQLDocStore:
    def __init__(self, get_pool: Callable):
        self.get_pool = get_pool

    async def put(self, doc_id: str, text: str, url: str):
        pool = self.get_pool()
        await pool.execute(
            "INSERT INTO raw_docs (doc_id,url,content) VALUES ($1,$2,$3) "
            "ON CONFLICT (doc_id) DO NOTHING",
            doc_id, url, text,
        )

    async def mget(self, doc_ids: list[str]) -> list[str]:
        pool = self.get_pool()
        rows = await pool.fetch(
            "SELECT content FROM raw_docs WHERE doc_id = ANY($1)", doc_ids
        )
        return [r["content"] for r in rows]
