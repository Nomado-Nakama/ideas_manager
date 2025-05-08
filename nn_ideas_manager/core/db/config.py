from os import getenv

from dotenv import load_dotenv

load_dotenv('/Projects/python/nakama-ideas-manager/configs/.env')

RESERVE_DB_DSN = getenv("RESERVE_DB_DSN", "postgres://postgres:postgres@localhost:5432/postgres")
DB_DSN = getenv("DATABASE_DSN", "postgres://postgres:postgres@db:5432/postgres")
POOL_MIN, POOL_MAX = 1, 10
