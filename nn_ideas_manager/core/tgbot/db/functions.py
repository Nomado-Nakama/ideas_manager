import pgcrud as pg
from psycopg import errors
from pgcrud import IdentifierExpression as i
from nn_ideas_manager.core.tgbot.config import DB_DSN


async def save_user_link(tg_id: int, url: str, status='unprocessed') -> None:
    q = pg.QueryBuilder.INSERT_INTO(
        i.user_links[i.url, i.telegram_id, i.status]
    ).VALUES((url, tg_id, status))

    # pgcrud’s async connection will accept a Query object
    async with await pg.async_connect(DB_DSN) as conn:
        try:
            await conn.execute(q)
        except errors.UniqueViolation:
            pass
