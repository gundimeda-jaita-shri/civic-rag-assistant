import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval.metrics import QueryEvalResult, evaluate_single_prediction
from services.rag_pipeline import CivicRAGService


def run_benchmark(
    dataset_path: Path = Path("eval/sample_queries.json"),
    top_k: int = 5,
) -> Dict:
    """Execute the multilingual RAG benchmark and return aggregate metrics."""
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    service = CivicRAGService()
    results: List[QueryEvalResult] = []

    for item in samples:
        outcome = service.ask(
            question=item["question"],
            target_language=item["language"],
            top_k=top_k,
        )
        scored = evaluate_single_prediction(
            query_id=item["id"],
            language=item["language"],
            retrieved_chunks=outcome.retrieved_chunks,
            response=outcome.response,
            expected_source=item["expected_source"],
            expected_page=int(item["expected_page"]),
            expected_keywords=item.get("expected_keywords", []),
            should_be_grounded=bool(item["should_be_grounded"]),
        )
        results.append(scored)

    total = max(len(results), 1)
    summary = {
        "total_queries": len(results),
        "mean_recall_at_k": round(sum(r.recall_at_k for r in results) / total, 4),
        "mrr": round(sum(r.reciprocal_rank for r in results) / total, 4),
        "grounding_accuracy": round(
            sum(1 for r in results if r.grounding_match) / total, 4
        ),
        "mean_keyword_coverage": round(
            sum(r.keyword_coverage for r in results) / total, 4
        ),
        "per_query": [r.__dict__ for r in results],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Civic RAG Evaluation Benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/sample_queries.json"),
        help="Path to JSON benchmark dataset",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth K")
    args = parser.parse_args()

    report = run_benchmark(dataset_path=args.dataset, top_k=args.top_k)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())