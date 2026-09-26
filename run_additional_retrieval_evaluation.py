"""Evaluate the frozen retriever on the additional hard-question challenge set."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from run_retrieval_evaluation import _evaluate, _print_rows, _rankings, _write_csv
from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.hybrid_retrieval import retrieve_candidates
from table_rag.retrieval_evaluation import load_evaluation_queries, query_result_rows
from table_rag.sparse_retrieval import load_bm25
from table_rag.summaries import sha256_file


FIXED_ALPHA = 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        help="JSON output (default: artifacts/additional_retrieval_evaluation.json)",
    )
    parser.add_argument(
        "--query-results-output", type=Path,
        help=("Per-query CSV output "
              "(default: artifacts/additional_retrieval_query_results.csv)"),
    )
    args = parser.parse_args()
    if args.rank_constant < 1:
        parser.error("--rank-constant must be positive")

    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    dataset = root / "dataset/generated_evaluation"
    queries_path = dataset / "additional_query_candidates.csv"
    qrels_path = dataset / "additional_qrels_candidates.csv"
    output_path = args.output or artifacts / "additional_retrieval_evaluation.json"
    query_results_path = (
        args.query_results_output
        or artifacts / "additional_retrieval_query_results.csv"
    )

    try:
        queries = load_evaluation_queries(queries_path, qrels_path)
        if not queries:
            raise ValueError("The additional challenge set is empty")
        if any(query.split != "test" for query in queries):
            raise ValueError("Every additional challenge query must use split=test")

        sparse_index = load_bm25(
            artifacts / "bm25_index.pkl",
            source_path=artifacts / "sparse_documents.jsonl",
        )
        dense_index = load_dense_index(
            artifacts / "dense_index.npz",
            summary_file_sha256=sha256_file(artifacts / "table_summaries.jsonl"),
        )
        model = load_embedding_model(local_files_only=args.local_files_only)

        retrieval_started = time.perf_counter()
        candidates = {
            query.query_id: retrieve_candidates(
                sparse_index, dense_index, model, query.question,
            )
            for query in queries
        }
        retrieval_seconds = time.perf_counter() - retrieval_started

        result_rows = [
            {
                "method": "RRF",
                "rank_constant": args.rank_constant,
                **_evaluate(
                    queries, candidates, "rrf",
                    rank_constant=args.rank_constant,
                ),
            },
            {
                "method": "BM25",
                "alpha": 0.0,
                **_evaluate(queries, candidates, "weighted", alpha=0.0),
            },
            {
                "method": f"Weighted a={FIXED_ALPHA:g}",
                "alpha": FIXED_ALPHA,
                **_evaluate(
                    queries, candidates, "weighted", alpha=FIXED_ALPHA,
                ),
            },
            {
                "method": "Dense",
                "alpha": 1.0,
                **_evaluate(queries, candidates, "weighted", alpha=1.0),
            },
        ]

        weighted_label = f"Weighted a={FIXED_ALPHA:g}"
        method_rankings = {
            "BM25": _rankings(candidates, "weighted", alpha=0.0),
            "Dense": _rankings(candidates, "weighted", alpha=1.0),
            "RRF": _rankings(
                candidates, "rrf", rank_constant=args.rank_constant,
            ),
            weighted_label: _rankings(
                candidates, "weighted", alpha=FIXED_ALPHA,
            ),
        }
        per_query_rows = query_result_rows(queries, method_rankings)
        _write_csv(query_results_path, per_query_rows)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    review_status_counts = {
        status: sum(query.review_status == status for query in queries)
        for status in sorted({query.review_status for query in queries})
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "additional_same_domain_challenge",
        "evaluation_role": "frozen diagnostic test; no parameter tuning",
        "review_status_counts": review_status_counts,
        "evaluation_sources": {
            "queries": {
                "path": str(queries_path.relative_to(root)),
                "sha256": sha256_file(queries_path),
            },
            "qrels": {
                "path": str(qrels_path.relative_to(root)),
                "sha256": sha256_file(qrels_path),
            },
        },
        "metrics": ["hit_rate@1", "hit_rate@3", "hit_rate@5", "mrr"],
        "alpha_definition": (
            "alpha*dense + (1-alpha)*BM25 after per-query min-max normalization"
        ),
        "fixed_alpha": FIXED_ALPHA,
        "alpha_selected_from": "original development set",
        "rank_constant": args.rank_constant,
        "candidate_retrieval_seconds": retrieval_seconds,
        "query_results": {
            "path": str(query_results_path.relative_to(root)),
            "sha256": sha256_file(query_results_path),
            "row_count": len(per_query_rows),
        },
        "challenge_results": result_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)

    _print_rows("Additional hard-question challenge", result_rows)
    print(f"\nFixed alpha from the original development set: {FIXED_ALPHA:g}")
    print("No tuning was performed on the additional challenge set.")
    print(f"Saved report: {output_path}")
    print(f"Saved per-query results: {query_results_path}")


if __name__ == "__main__":
    main()
