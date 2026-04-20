"""Configuration management module for the FastAPI diamond chat application.

This module provides centralized configuration settings using Pydantic BaseSettings,
with support for environment variables and .env file loading.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os


load_dotenv()


class Settings(BaseSettings):
    """Application configuration settings.
    
    Attributes:
        DATABASE_URL (str): PostgreSQL async database connection URL.
            Defaults to local PostgreSQL instance.
        NVIDIA_API_KEY (str | None): API key for NVIDIA AI endpoints (Gemma model).
        HOST (str): Server host address. Defaults to "0.0.0.0" (all interfaces).
        PORT (int): Server port number. Defaults to 8000.
        CORS_ALLOW_ORIGINS (str): Comma-separated list of allowed CORS origins.
            Includes localhost development servers on ports 3000, 5173, and 5500.
        CORS_ALLOW_ORIGIN_REGEX (str): Regex pattern for dynamic CORS origin matching.
            Allows localhost and 127.0.0.1 on any port.
        CORS_ALLOW_METHODS (str): Comma-separated HTTP methods allowed for CORS.
        CORS_ALLOW_HEADERS (str): Headers allowed in CORS requests. Wildcard "*" allows all.
        CORS_ALLOW_CREDENTIALS (bool): Whether to allow credentials in CORS requests.
    """
    DATABASE_URL: str = "postgresql+asyncpg://postgres:0207@localhost:5432/postgres"
    NVIDIA_API_KEY: str | None = None
    NVIDIA_TOOL_MODEL: str = "google/gemma-4-31b-it"
    NVIDIA_CHAT_MODEL: str = "google/gemma-4-31b-it"
    NVIDIA_TEMPERATURE: float = 0.3
    NVIDIA_TOP_P: float = 0.9
    NVIDIA_TOOL_MAX_TOKENS: int = 1200
    NVIDIA_CHAT_MAX_TOKENS: int = 512
    AGENT_TIMEOUT_SECONDS: int = 45
    CHAT_TIMEOUT_SECONDS: int = 25
    AGENT_MAX_ITERATIONS: int = 3
    AGENT_VERBOSE: bool = False
    MEMORY_MAX_MESSAGES: int = 16
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
        """Pydantic config for loading environment variables from .env file."""
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings instance.
    
    Uses LRU caching to ensure only one Settings instance is created and reused
    throughout the application lifecycle.
    
    Returns:
        Settings: Singleton instance of application settings.
    """
    return Settings()
