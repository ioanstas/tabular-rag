"""Search a saved BM25 index without rebuilding it."""
import argparse
from pathlib import Path
from table_rag.sparse_retrieval import load_bm25, search_bm25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Quote the complete question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    artifacts = Path(__file__).resolve().parent / "artifacts"
    try:
        index = load_bm25(
            artifacts / "bm25_index.pkl",
            source_path=artifacts / "sparse_documents.jsonl",
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Loaded BM25 index with {len(index.table_ids)} table documents.")
    results = search_bm25(index, args.question, args.top_k)
    if not results:
        print("No matching query terms found.")
    for result in results:
        print(f"{result.rank}. {result.table_id}  score={result.score:.4f}")


if __name__ == "__main__":
    main()
