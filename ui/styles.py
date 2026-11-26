from typing import Dict, List

SUPPORTED_LANGUAGES: List[str] = [
    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Marathi",
    "Bengali",
]

CONFIDENCE_BADGES: Dict[str, str] = {
    "High": "🟢 High Confidence",
    "Medium": "🟡 Medium Confidence",
    "Low": "🟠 Low Confidence",
    "Insufficient Evidence": "🔴 Insufficient Evidence",
}


def format_confidence_badge(confidence_level: str) -> str:
    """Return an accessible status badge for a model confidence tier."""
    return CONFIDENCE_BADGES.get(confidence_level, f"⚪ {confidence_level}")


def inject_civic_styles() -> str:
    """Return institutional slate & navy CSS for the Streamlit Civic Registry UI."""
    return """
    <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        .civic-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: #f8fafc;
            padding: 1.35rem 1.6rem;
            border-radius: 10px;
            margin-bottom: 1.25rem;
            border-left: 6px solid #f59e0b;
        }
        .civic-header h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.65rem;
            font-weight: 700;
            color: #ffffff;
        }
        .civic-header p {
            margin: 0;
            font-size: 0.95rem;
            color: #cbd5e1;
        }
        .citation-box {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #1e3a8a;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
        }
        .citation-meta {
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .citation-quote {
            color: #334155;
            font-style: italic;
        }
    </style>
    """