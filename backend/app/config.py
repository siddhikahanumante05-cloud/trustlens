"""Configuration settings for TrustLens backend."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    API_TOKEN: str = ""
    SECRET_KEY: str = "trustlens-insecure-secret-change-me"
    USE_MOCK: bool = False
    CLIP_MODE: bool = False
    FPS: int = 15
    WINDOW_SEC: int = 3
    STRIDE_SEC: int = 1
    MAX_SESSIONS: int = 4
    DEMO_RUN_ALL: bool = True
    KEEP_RAW: bool = False
    CASE_TTL_HOURS: int = 24
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    NARRATOR_MODE: str = "auto"  # auto | template | llm
    MEMORY_PROFILE: str = "8gb"
    TAP: str = "agent"  # agent | caller
    LLM_TIMEOUT_SEC: float = 8.0
    DEVICE: str = "cpu"
    ORT_THREADS: int = 4
    DEVICE_BUDGET: str = "low"  # low | high
    DEMO_BYPASS_CONSENT: bool = False
    ENV: str = "development"  # demo | development | production

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()
