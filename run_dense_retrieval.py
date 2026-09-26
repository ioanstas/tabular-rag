"""Search the persisted dense index without re-encoding table summaries."""
import argparse
from pathlib import Path

from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.dense_retrieval import search_dense
from table_rag.summaries import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Quote the complete question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use cached model files without downloading")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not args.question.strip():
        parser.error("question must be non-empty text")
    artifacts = Path(__file__).resolve().parent / "artifacts"
    try:
        index = load_dense_index(
            artifacts / "dense_index.npz",
            summary_file_sha256=sha256_file(artifacts / "table_summaries.jsonl"),
        )
        model = load_embedding_model(local_files_only=args.local_files_only)
        results = search_dense(index, model, args.question, args.top_k)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Loaded dense index with {len(index.table_ids)} table summaries.")
    for result in results:
        print(f"{result.rank}. {result.table_id}  score={result.score:.4f}")


if __name__ == "__main__":
    main()
