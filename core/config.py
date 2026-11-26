from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    sqlite_db_path: Path = Field(default=Path("data/civic_rag.db"), alias="SQLITE_DB_PATH")
    faiss_index_dir: Path = Field(default=Path("data/faiss_index"), alias="FAISS_INDEX_DIR")
    upload_dir: Path = Field(default=Path("data/uploads"), alias="UPLOAD_DIR")

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    max_upload_mb: int = Field(default=50, alias="MAX_UPLOAD_MB")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()