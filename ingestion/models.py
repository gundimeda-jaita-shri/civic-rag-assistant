from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExtractedPage:
    """Represents text and Markdown-serialized tables extracted from a single PDF page."""

    page_number: int
    raw_text: str
    markdown_tables: List[str] = field(default_factory=list)

    @property
    def combined_content(self) -> str:
        sections: List[str] = []
        if self.raw_text.strip():
            sections.append(self.raw_text.strip())
        for idx, table in enumerate(self.markdown_tables, start=1):
            sections.append(f"[TABLE {idx} ON PAGE {self.page_number}]\n{table}")
        return "\n\n".join(sections)


@dataclass
class ParsedDocument:
    """Container for all extracted pages and metadata from a civic PDF."""

    filename: str
    file_hash: str
    page_count: int
    table_count: int
    pages: List[ExtractedPage]
    metadata: Dict[str, Any] = field(default_factory=dict)