from nn_ideas_manager.core.tgbot.db.connection import get_pool
from asyncpg import Record


async def mark_link_status(link_id: str, new_status: str) -> None:
    """
    Update a single link’s status.
    """
    sql = "UPDATE user_links SET status = $1 WHERE id = $2"
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(sql, new_status, link_id)


async def fetch_user_links_by_status(status: str) -> list[Record]:
    """
    Returns all rows from user_links with the given status.
    """
    sql = """
        SELECT id, url, telegram_id
          FROM user_links
         WHERE status = $1
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, status)
    return rows


async def save_telegram_user(user: dict) -> None:
    """
    INSERT a new user or skip if telegram_id already exists.
    """
    sql = """
        INSERT INTO telegram_users (telegram_id, first_name, last_name, role)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (telegram_id) DO NOTHING;
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            user["id"],
            user.get("first_name"),
            user.get("last_name"),
            "noone",       # default role
        )


async def get_telegram_user_role(tg_id: int) -> str | None:
    sql = "SELECT role FROM telegram_users WHERE telegram_id = $1;"
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, tg_id)
        return row["role"] if row else None


async def save_user_link(tg_id: int, url: str, status: str = "unprocessed") -> None:
    sql = """
        INSERT INTO user_links (url, telegram_id, status)
        VALUES ($1, $2, $3)
        ON CONFLICT (url) DO NOTHING;
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(sql, url, tg_id, status)

