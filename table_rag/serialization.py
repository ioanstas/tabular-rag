"""Represent complete source tables as deterministic searchable documents."""
import json
from pathlib import Path

from table_rag.ingestion import LoadedTable


def serialize_table(table: LoadedTable) -> str:
    """Preserve ingestion's text values and row/column order using JSON lines."""
    fields = [
        ("table_id", table.table_id),
        ("title", table.metadata["title"]),
        ("domain", table.metadata["domain"]),
        ("columns", table.data.columns.tolist()),
    ]
    lines = [f"{name}: {json.dumps(value, ensure_ascii=False)}" for name, value in fields]
    for row in table.data.itertuples(index=False, name=None):
        lines.append(f"row: {json.dumps(list(row), ensure_ascii=False)}")
    return "\n".join(lines)


def serialize_tables(tables: dict[str, LoadedTable]) -> list[dict[str, str]]:
    """Return one document per table in dictionary insertion order."""
    documents = []
    for table_id, table in tables.items():
        if table_id != table.table_id:
            raise ValueError(
                f"Dictionary key {table_id!r} does not match table ID {table.table_id!r}"
            )
        documents.append({"table_id": table_id, "serialized_text": serialize_table(table)})
    return documents


def save_documents(
    documents: list[dict[str, str]], output_path: str | Path,
) -> None:
    """Validate all documents before replacing a UTF-8 JSONL artifact."""
    records = []
    seen = set()
    for document in documents:
        record = {}
        for key in ("table_id", "serialized_text"):
            value = document.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Document {key!r} must be a non-empty string")
            record[key] = value
        if record["table_id"] in seen:
            raise ValueError(f"Duplicate table_id: {record['table_id']!r}")
        seen.add(record["table_id"])
        records.append(json.dumps(record, ensure_ascii=False) + "\n")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(records), encoding="utf-8", newline="\n")
