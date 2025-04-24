from os import getenv
from pathlib import Path

from knowledge_devourer.core.config import Config

BOT_TOKEN = getenv("BOT_TOKEN", "7944366998:AAEHYuBZFd59fzbm68MmGSdMNh9n2Z39DdE")
DB_DSN = getenv("DATABASE_DSN", "postgres://postgres:postgres@localhost:5432/postgres")
POOL_MIN, POOL_MAX = 1, 10

kd_config = Config(
    base_dir=Path(getenv("KD_STORAGE_FULL_PATH", r"E:\Projects\python\knowledge_devourer\storage"))
)