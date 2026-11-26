import argparse
import json
import sys
from pathlib import Path

from core.exceptions import CivicRAGError
from services.document_manager import CivicDocumentManager
from services.rag_pipeline import CivicRAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="civic-rag",
        description="Multilingual Civic Scheme & Document Assistant CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest command
    ingest_cmd = subparsers.add_parser("ingest", help="Ingest and index a civic PDF")
    ingest_cmd.add_argument("pdf_path", type=Path, help="Path to the PDF file")

    # ask command
    ask_cmd = subparsers.add_parser("ask", help="Ask a grounded civic question")
    ask_cmd.add_argument("question", type=str, help="Question to answer")
    ask_cmd.add_argument(
        "--lang",
        type=str,
        default="English",
        help="Target response language (default: English)",
    )
    ask_cmd.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional filename filter (e.g., scheme.pdf)",
    )
    ask_cmd.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of hybrid evidence chunks to retrieve",
    )
    ask_cmd.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON response",
    )

    # list command
    subparsers.add_parser("list", help="List all indexed civic documents")

    # stats command
    subparsers.add_parser("stats", help="Display SQLite registry & latency telemetry")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        service = CivicRAGService()
        manager = CivicDocumentManager(db=service.db, retriever=service.retriever)

        if args.command == "ingest":
            pdf_path: Path = args.pdf_path
            if not pdf_path.exists():
                print(f"Error: File not found -> {pdf_path}", file=sys.stderr)
                return 1
            managed_path = manager.save_uploaded_bytes(
                pdf_path.name, pdf_path.read_bytes()
            )
            outcome = service.ingest_pdf(managed_path)
            print(
                f"[{outcome.status.upper()}] {outcome.filename} | "
                f"Pages: {outcome.page_count} | Tables: {outcome.table_count} | "
                f"Chunks: {outcome.chunk_count} | SHA256: {outcome.file_hash[:12]}..."
            )
            return 0

        if args.command == "ask":
            outcome = service.ask(
                question=args.question,
                target_language=args.lang,
                top_k=args.top_k,
                source_filter=args.source,
            )
            resp = outcome.response
            if args.json:
                print(
                    json.dumps(
                        {
                            "question": outcome.question,
                            "latency_ms": outcome.latency_ms,
                            "response": resp.model_dump(),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return 0

            print(f"\nAnswer ({resp.target_language}):\n{resp.answer}\n")
            print(
                f"Status: {'Grounded' if resp.is_grounded else 'Insufficient Evidence'} | "
                f"Confidence: {resp.confidence_level} | Latency: {outcome.latency_ms} ms"
            )
            if resp.citations:
                print("\nCitations:")
                for idx, cite in enumerate(resp.citations, start=1):
                    print(f"  {idx}. [{cite.source}, p.{cite.page}] \"{cite.quote}\"")
            if resp.missing_information:
                print(f"\nCaveats: {resp.missing_information}")
            return 0

        if args.command == "list":
            docs = service.list_indexed_documents()
            if not docs:
                print("No civic documents indexed yet.")
                return 0
            for doc in docs:
                print(
                    f"- {doc['filename']} | Pages: {doc['page_count']} | "
                    f"Chunks: {doc['chunk_count']} | Tables: {doc['table_count']} | "
                    f"Uploaded: {doc['uploaded_at']}"
                )
            return 0

        if args.command == "stats":
            stats = service.get_stats()
            print(json.dumps(stats, indent=2))
            return 0

    except CivicRAGError as exc:
        print(f"CivicRAG Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())