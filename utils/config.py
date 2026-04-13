from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os


load_dotenv()


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:0207@localhost:5432/postgres"
    NVIDIA_API_KEY: str | None = None
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ALLOW_ORIGINS: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5500,http://127.0.0.1:5500"
    )
    # Matches local frontend origins on any port across developer machines.
    CORS_ALLOW_ORIGIN_REGEX: str = r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$"
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    CORS_ALLOW_HEADERS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
