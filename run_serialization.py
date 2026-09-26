"""Generate sparse documents using paths relative to this script."""
from pathlib import Path

from table_rag.ingestion import load_all_tables, load_manifest
from table_rag.serialization import save_documents, serialize_tables


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "dataset"
    manifest = load_manifest(dataset_dir / "table_manifest.csv")
    documents = serialize_tables(load_all_tables(manifest, dataset_dir))
    output_path = project_dir / "artifacts" / "sparse_documents.jsonl"
    save_documents(documents, output_path)
    if documents:
        print("First document preview:")
        print(documents[0]["serialized_text"][:800])
    print(f"Saved {len(documents)} documents to {output_path}")


if __name__ == "__main__":
    main()
