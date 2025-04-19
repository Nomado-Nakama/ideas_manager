from os import getenv

BOT_TOKEN = getenv("BOT_TOKEN")
DB_DSN = getenv("DATABASE_DSN")
POOL_MIN, POOL_MAX = 1, 10
