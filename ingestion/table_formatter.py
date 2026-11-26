from typing import List, Optional


def _clean_cell(cell: Optional[str]) -> str:
    """Normalize whitespace and escape pipe characters inside a table cell."""
    if cell is None:
        return ""
    cleaned = " ".join(str(cell).split())
    return cleaned.replace("|", "\\|")


def rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Convert extracted 2D table rows into a clean GitHub-Flavored Markdown table."""
    if not rows:
        return ""

    normalized: List[List[str]] = []
    for row in rows:
        cleaned_row = [_clean_cell(cell) for cell in row]
        if any(cleaned_row):
            normalized.append(cleaned_row)

    if len(normalized) < 2:
        return ""

    col_count = max(len(r) for r in normalized)
    padded = [r + [""] * (col_count - len(r)) for r in normalized]

    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in padded[1:]]

    return "\n".join([header, separator, *body_lines])