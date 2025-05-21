import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )
    PRODUCTION: bool = os.getenv("ENV") == "production"
    TELEMETRY: bool = os.getenv("TEL", "false").lower() == "true"
    HUGGINGFACE_APIKEY: str = os.getenv("HUGGINGFACE_APIKEY")


settings = Settings()
