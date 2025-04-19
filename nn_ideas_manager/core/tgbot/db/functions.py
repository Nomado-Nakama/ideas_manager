import loguru
import pgcrud as pg
from psycopg import errors
from nn_ideas_manager.core.tgbot.config import DB_DSN
from pgcrud import IdentifierExpression as i, QueryBuilder as q


async def save_telegram_user(tg_user: dict) -> None:
    """
    Upsert a user into telegram_users; ignore if already exists.
    """
    async with await pg.async_connect(DB_DSN) as conn:
        try:
            await pg.async_insert_one(
                cursor=conn,
                insert_into=i.telegram_users[
                    i.telegram_id, i.first_name, i.last_name, i.role
                ],
                values=(
                    tg_user["id"],
                    tg_user.get("first_name"),
                    tg_user.get("last_name"),
                    "noone",
                )
            )
        except errors.UniqueViolation:
            pass


async def get_telegram_user_role(tg_id: int) -> str | None:
    """
    Fetch the stored role for a given telegram_id, or None if not found.
    """
    q = (
        pg.QueryBuilder.SELECT(i.role)
        .FROM(i.telegram_users)
        .WHERE(i.telegram_id == tg_id)
    )
    async with await pg.async_connect(DB_DSN) as conn:
        row: pg.AsyncCursor = await conn.execute(q)
        loguru.logger.info(row.pgresult.get_value(row_number=1, column_number=4))
        return row.pgresult.get_value(row_number=1, column_number=4) if row.pgresult else None


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
