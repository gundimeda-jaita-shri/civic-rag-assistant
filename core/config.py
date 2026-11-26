import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    sqlite_db_path: Path = Path(os.getenv("SQLITE_DB_PATH", "data/civic_rag.db"))
    faiss_index_dir: Path = Path(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "750"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    def ensure_directories(self) -> None:
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()