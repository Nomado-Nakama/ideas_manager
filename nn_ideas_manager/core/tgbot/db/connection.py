import asyncpg
from nn_ideas_manager.core.tgbot.config import DB_DSN, POOL_MIN, POOL_MAX

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=POOL_MIN,
            max_size=POOL_MAX,
        )
    return _pool
