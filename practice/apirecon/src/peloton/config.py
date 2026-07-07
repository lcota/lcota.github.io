"""Centralized settings for the Peloton API client."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and/or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="PELOTON_",  # TODO: init.sh substitutes this (e.g. PELOTON_)
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    email: str = Field(default="", description="Account email")
    password: str = Field(default="", description="Account password")
    base_url: str = Field(
        default="https://api.onepeloton.com/v1",  # TODO: set to real base URL
        description="API base URL",
    )
    token_cache: Path = Field(
        default=Path.home() / ".peloton" / "token.json",
        description="Path to the cached token file",
    )

    # Request timeouts in seconds
    request_timeout: float = Field(default=30.0, description="HTTP request timeout")
    login_timeout: float = Field(
        default=60.0, description="Login request timeout in seconds"
    )


settings = Settings()
