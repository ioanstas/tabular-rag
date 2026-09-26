"""Run source-table ingestion using the dataset beside this script."""
from pathlib import Path

from table_rag.ingestion import load_all_tables, load_manifest


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    manifest = load_manifest(dataset_dir / "table_manifest.csv")
    tables = load_all_tables(manifest, dataset_dir)
    for table_id, table in tables.items():
        rows, columns = table.data.shape
        print(f"{table_id}: {rows} rows, {columns} columns")
    print(f"Successfully loaded and validated {len(tables)} tables.")


if __name__ == "__main__":
    main()
