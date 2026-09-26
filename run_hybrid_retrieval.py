"""Search saved BM25 and dense indexes with RRF, weighted fusion, or both."""
import argparse
import math
from pathlib import Path

from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.hybrid_retrieval import fuse_rrf, fuse_weighted, retrieve_candidates
from table_rag.sparse_retrieval import load_bm25
from table_rag.summaries import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Quote the complete question")
    parser.add_argument("--method", choices=("rrf", "weighted", "both"), default="rrf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dense weight for weighted fusion, from 0 to 1 (default: 0.5)")
    parser.add_argument("--rank-constant", type=int, default=60,
                        help="RRF smoothing constant (default: 60)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use cached model files without downloading")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must be non-empty text")
    if args.top_k < 1 or args.rank_constant < 1:
        parser.error("--top-k and --rank-constant must be positive")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    artifacts = Path(__file__).resolve().parent / "artifacts"
    try:
        sparse_index = load_bm25(
            artifacts / "bm25_index.pkl",
            source_path=artifacts / "sparse_documents.jsonl",
        )
        dense_index = load_dense_index(
            artifacts / "dense_index.npz",
            summary_file_sha256=sha256_file(artifacts / "table_summaries.jsonl"),
        )
        if set(sparse_index.table_ids) != set(dense_index.table_ids):
            raise ValueError("Sparse and dense indexes must cover the same table IDs")
        model = load_embedding_model(local_files_only=args.local_files_only)
        sparse, dense = retrieve_candidates(sparse_index, dense_index, model, args.question)
        outputs = []
        if args.method in ("rrf", "both"):
            outputs.append((
                f"RRF (rank_constant={args.rank_constant})",
                fuse_rrf(sparse, dense, args.top_k, rank_constant=args.rank_constant),
            ))
        if args.method in ("weighted", "both"):
            outputs.append((
                f"Weighted (alpha={args.alpha:g}, normalization=min-max)",
                fuse_weighted(sparse, dense, args.top_k, alpha=args.alpha),
            ))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Loaded indexes covering {len(dense_index.table_ids)} tables.")
    print(f"Candidates: {len(sparse)} BM25 matches, {len(dense)} dense results.")
    for label, results in outputs:
        print(f"\n{label}")
        if not results:
            print("No candidates from the active retriever.")
        for result in results:
            print(f"{result.rank}. {result.table_id}  score={result.score:.6f}")


if __name__ == "__main__":
    main()
