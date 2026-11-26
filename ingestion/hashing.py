import hashlib
from pathlib import Path
from core.exceptions import DocumentIngestionError

ALLOWED_EXTENSIONS = {".pdf"}


def compute_sha256(file_path: Path) -> str:
    """Compute a deterministic SHA-256 checksum for a PDF file."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise DocumentIngestionError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_pdf_file(file_path: Path, max_size_mb: int = 50) -> None:
    """Verify that the uploaded file exists, is a PDF, is non-empty, and fits size bounds."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise DocumentIngestionError(f"Document does not exist: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise DocumentIngestionError(
            f"Unsupported file format '{path.suffix}'. Only PDF documents are accepted."
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise DocumentIngestionError(f"Uploaded PDF '{path.name}' is empty (0 bytes).")

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentIngestionError(
            f"PDF '{path.name}' exceeds the {max_size_mb} MB institutional upload limit."
        )