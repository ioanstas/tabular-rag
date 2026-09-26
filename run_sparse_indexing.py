"""Build and save the offline BM25 artifact from serialized documents."""
from pathlib import Path
from table_rag.sparse_retrieval import (
    bm25_metadata, build_bm25, load_sparse_documents, save_bm25, source_sha256,
)


def main() -> None:
    artifacts = Path(__file__).resolve().parent / "artifacts"
    source_path = artifacts / "sparse_documents.jsonl"
    output_path = artifacts / "bm25_index.pkl"
    metadata = bm25_metadata(source_path)
    documents = load_sparse_documents(source_path)
    index = build_bm25(documents)
    if source_sha256(source_path) != metadata["source_sha256"]:
        raise ValueError("Source documents changed during indexing; run indexing again.")
    save_bm25(index, metadata, output_path)
    print(f"Saved BM25 index with {len(index.table_ids)} table documents to {output_path}")


if __name__ == "__main__":
    main()
