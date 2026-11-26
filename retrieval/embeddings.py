from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings
from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the multilingual HuggingFace sentence-transformer model."""
    settings = get_settings()
    model_name = settings.embedding_model

    try:
        logger.info("Loading multilingual embedding model '%s'...", model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RetrievalError(
            f"Failed to initialize embedding model '{model_name}': {exc}"
        ) from exc