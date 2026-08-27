"""
Application configuration.

All runtime configuration is sourced from environment variables (optionally
loaded from a .env file via python-dotenv). Nothing here should hardcode
secrets. See backend/.env.example for the full list of supported variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "RepoCrawl"
    environment: str = "development"
    log_level: str = "INFO"

    # --- CORS ---
    # Comma-separated list of allowed origins. Defaults cover local Vite dev.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Hugging Face / LLM ---
    huggingfacehub_api_token: str = ""
    hf_token: str = ""
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage / runtime data (never committed) ---
    chroma_persist_dir: str = "./runtime_data/chroma_db"
    repo_clone_dir: str = "./runtime_data/repos"

    # --- Retrieval / chat tuning ---
    max_history_turns: int = 6

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def hf_api_token(self) -> str:
        """Prefer HUGGINGFACEHUB_API_TOKEN, fall back to HF_TOKEN."""
        return self.huggingfacehub_api_token or self.hf_token

    def ensure_runtime_dirs(self) -> None:
        """Create runtime data directories if they don't already exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.repo_clone_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
