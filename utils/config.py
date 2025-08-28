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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
