"""Tune weighted hybrid retrieval on dev, then report held-out test results."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.hybrid_retrieval import fuse_rrf, fuse_weighted, retrieve_candidates
from table_rag.retrieval_evaluation import (
    load_evaluation_queries, query_result_rows, retrieval_metrics, select_alpha,
)
from table_rag.sparse_retrieval import load_bm25
from table_rag.summaries import sha256_file


DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EXPOSED_TEST_IDS = frozenset({"q002", "q003"})


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if (not values or any(not math.isfinite(alpha) or not 0 <= alpha <= 1
                          for alpha in values)):
        raise argparse.ArgumentTypeError("every alpha must be between 0 and 1")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("alphas must be unique")
    return values


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rankings(candidates, method: str, *, alpha=None, rank_constant=60):
    if method == "rrf":
        return {
            query_id: fuse_rrf(
                sparse, dense, top_k=len(dense), rank_constant=rank_constant,
            )
            for query_id, (sparse, dense) in candidates.items()
        }
    return {
        query_id: fuse_weighted(sparse, dense, top_k=len(dense), alpha=alpha)
        for query_id, (sparse, dense) in candidates.items()
    }


def _evaluate(queries, candidates, method: str, **settings):
    selected = {query.query_id: candidates[query.query_id] for query in queries}
    started = time.perf_counter()
    metrics = retrieval_metrics(queries, _rankings(selected, method, **settings))
    metrics["fusion_seconds"] = time.perf_counter() - started
    return metrics


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'method':<20} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'MRR':>8} {'queries':>8}")
    for row in rows:
        print(
            f"{row['method']:<20} {row['hit_rate@1']:>8.3f} "
            f"{row['hit_rate@3']:>8.3f} {row['hit_rate@5']:>8.3f} "
            f"{row['mrr']:>8.3f} {row['query_count']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphas", type=_parse_alphas, default=DEFAULT_ALPHAS,
        help="comma-separated dense weights (default: 0,.2,.4,.6,.8,1)",
    )
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--include-exposed-test", action="store_true",
        help="include q002/q003 in test (then it is not untouched)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="JSON output (default: artifacts/retrieval_evaluation.json)",
    )
    args = parser.parse_args()
    if args.rank_constant < 1:
        parser.error("--rank-constant must be positive")
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    dataset = root / "dataset/generated_evaluation"
    output_path = args.output or artifacts / "retrieval_evaluation.json"
    try:
        queries_path = dataset / "query_candidates.csv"
        qrels_path = dataset / "qrels_candidates.csv"
        queries = load_evaluation_queries(queries_path, qrels_path)
        dev = [query for query in queries if query.split == "dev"]
        test = [
            query for query in queries
            if query.split == "test"
            and (args.include_exposed_test or query.query_id not in EXPOSED_TEST_IDS)
        ]
        review_status_counts = {
            status: sum(query.review_status == status for query in queries)
            for status in sorted({query.review_status for query in queries})
        }
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
            for query in dev + test
        }
        retrieval_seconds = time.perf_counter() - retrieval_started

        dev_rows = []
        rrf_metrics = _evaluate(
            dev, candidates, "rrf", rank_constant=args.rank_constant,
        )
        dev_rows.append({
            "method": "RRF", "rank_constant": args.rank_constant, **rrf_metrics,
        })
        for alpha in args.alphas:
            metrics = _evaluate(dev, candidates, "weighted", alpha=alpha)
            label = (
                "BM25" if alpha == 0 else
                "Dense" if alpha == 1 else
                f"Weighted a={alpha:g}"
            )
            dev_rows.append({"method": label, "alpha": alpha, **metrics})
        weighted_rows = [row for row in dev_rows if "alpha" in row]
        selected_alpha = select_alpha(weighted_rows)

        test_rows = [{
            "method": "RRF",
            "rank_constant": args.rank_constant,
            **_evaluate(test, candidates, "rrf", rank_constant=args.rank_constant),
        }]
        test_alphas = [
            (0.0, "BM25"),
            (selected_alpha, f"Weighted a={selected_alpha:g}"),
            (1.0, "Dense"),
        ]
        seen_alphas = set()
        for alpha, label in test_alphas:
            if alpha in seen_alphas:
                continue
            seen_alphas.add(alpha)
            test_rows.append({
                "method": label,
                "alpha": alpha,
                **_evaluate(test, candidates, "weighted", alpha=alpha),
            })

        weighted_label = f"Weighted a={selected_alpha:g}"
        method_rankings = {
            "BM25": _rankings(candidates, "weighted", alpha=0.0),
            "Dense": _rankings(candidates, "weighted", alpha=1.0),
            "RRF": _rankings(
                candidates, "rrf", rank_constant=args.rank_constant,
            ),
            weighted_label: _rankings(
                candidates, "weighted", alpha=selected_alpha,
            ),
        }
        query_results = query_result_rows(dev + test, method_rankings)
        query_results_path = artifacts / "retrieval_query_results.csv"
        _write_csv(query_results_path, query_results)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_status": (
            "human_approved"
            if set(review_status_counts) == {"human_approved"}
            else "mixed"
        ),
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
        "alphas_tuned_on": "dev",
        "selected_alpha": selected_alpha,
        "rank_constant": args.rank_constant,
        "excluded_exposed_test_ids": (
            [] if args.include_exposed_test else sorted(EXPOSED_TEST_IDS)
        ),
        "candidate_retrieval_seconds": retrieval_seconds,
        "query_results": {
            "path": str(query_results_path.relative_to(root)),
            "sha256": sha256_file(query_results_path),
            "row_count": len(query_results),
        },
        "dev_results": dev_rows,
        "test_results": test_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)

    if review_status_counts.get("pending_review"):
        print("WARNING: evaluation labels are pending_review; results are provisional.")
    _print_rows("Development sweep", dev_rows)
    print(f"\nSelected alpha by development MRR: {selected_alpha:g}")
    _print_rows("Held-out test comparison", test_rows)
    excluded = report["excluded_exposed_test_ids"]
    print(f"\nExcluded exposed test IDs: {', '.join(excluded) if excluded else 'none'}")
    print(f"Saved report: {output_path}")
    print(f"Saved per-query results: {query_results_path}")


if __name__ == "__main__":
    main()
