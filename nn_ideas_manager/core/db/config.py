from os import getenv

DB_DSN = getenv("DATABASE_DSN", "postgres://postgres:postgres@localhost:5432/postgres")
POOL_MIN, POOL_MAX = 1, 10
