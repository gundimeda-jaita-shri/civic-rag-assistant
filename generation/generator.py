from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings
from core.exceptions import GenerationError
from core.logger import get_logger
from generation.prompts import build_civic_prompt_template, format_retrieved_chunks
from generation.schemas import GroundedCivicResponse
from retrieval.hybrid_retriever import RetrievedChunk

logger = get_logger(__name__)


class CivicGroundedGenerator:
    """Synthesizes citation-backed multilingual answers using Google Gemini."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.gemini_model
        self.api_key = api_key if api_key is not None else settings.google_api_key
        self.temperature = temperature
        self.prompt = build_civic_prompt_template()

    def _build_structured_llm(self):
        if not self.api_key:
            raise GenerationError(
                "GOOGLE_API_KEY is missing. Set it in your .env file before querying Gemini."
            )
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
        )
        return llm.with_structured_output(GroundedCivicResponse)

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        target_language: str = "English",
    ) -> GroundedCivicResponse:
        """Generate a schema-validated, citation-grounded response from retrieved chunks."""
        clean_question = question.strip()
        if not clean_question:
            raise GenerationError("Question cannot be empty.")

        if not chunks:
            logger.info("No chunks retrieved; returning deterministic ungrounded fallback.")
            return GroundedCivicResponse(
                answer=(
                    "No relevant scheme documents were found in the knowledge base to answer "
                    "this question. Please upload the official scheme PDF first."
                ),
                target_language=target_language,
                is_grounded=False,
                confidence_level="Insufficient Evidence",
                citations=[],
                missing_information="No matching indexed document chunks available.",
            )

        formatted_context = format_retrieved_chunks(chunks)

        try:
            structured_llm = self._build_structured_llm()
            chain = self.prompt | structured_llm
            result: GroundedCivicResponse = chain.invoke(
                {
                    "target_language": target_language,
                    "question": clean_question,
                    "formatted_context": formatted_context,
                }
            )
            logger.info(
                "Generated response (lang=%s, grounded=%s, confidence=%s, citations=%d).",
                result.target_language,
                result.is_grounded,
                result.confidence_level,
                len(result.citations),
            )
            return result
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Gemini generation failed: {exc}") from exc