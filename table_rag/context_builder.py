"""Resolve retrieved table IDs to complete, provenance-preserving CSV context."""
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

from table_rag.ingestion import load_manifest, load_table
from table_rag.sparse_retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextTable:
    """One retrieved source table prepared for answer generation."""

    table_id: str
    title: str
    domain: str
    source_path: str
    retrieval_rank: int
    retrieval_score: float
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ContextPackage:
    """The exact evidence and rendered text passed to an answer model."""

    question: str
    tables: tuple[ContextTable, ...]
    context_text: str

    @property
    def table_ids(self) -> tuple[str, ...]:
        return tuple(table.table_id for table in self.tables)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_results(results: Sequence[RetrievalResult]) -> None:
    if not results:
        raise ValueError("At least one retrieval result is required")
    seen: set[str] = set()
    previous_score = math.inf
    for expected_rank, result in enumerate(results, start=1):
        if not isinstance(result.table_id, str) or not result.table_id.strip():
            raise ValueError("Retrieval results require non-empty table IDs")
        if result.table_id in seen:
            raise ValueError(f"Duplicate retrieval table ID: {result.table_id}")
        if result.rank != expected_rank:
            raise ValueError("Retrieval ranks must be consecutive and start at 1")
        if not math.isfinite(result.score) or result.score > previous_score:
            raise ValueError("Retrieval scores must be finite and descending")
        seen.add(result.table_id)
        previous_score = result.score


def _safe_source_path(entry, dataset_dir: Path) -> Path:
    dataset_root = dataset_dir.resolve()
    source_path = (dataset_root / str(entry["file_path"])).resolve()
    if source_path != dataset_root and dataset_root not in source_path.parents:
        raise ValueError(
            f"Table '{entry['table_id']}' resolves outside the dataset directory"
        )
    return source_path


def load_context_tables(
    retrieval_results: Sequence[RetrievalResult],
    manifest_path: str | Path,
    dataset_dir: str | Path,
    *,
    top_k: int = 3,
) -> tuple[ContextTable, ...]:
    """Load the complete original CSVs for the highest-ranked table IDs."""
    _positive_int(top_k, "top_k")
    _validate_results(retrieval_results)
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(manifest_path)
    manifest_by_id = {
        str(entry["table_id"]): entry for _, entry in manifest.iterrows()
    }

    context_tables = []
    for result in retrieval_results[:top_k]:
        entry = manifest_by_id.get(result.table_id)
        if entry is None:
            raise ValueError(
                f"Retrieved table ID is absent from the manifest: {result.table_id}"
            )
        _safe_source_path(entry, dataset_dir)
        loaded = load_table(entry, dataset_dir)
        columns = tuple(str(column) for column in loaded.data.columns)
        rows = tuple(
            tuple(str(value) for value in row)
            for row in loaded.data.itertuples(index=False, name=None)
        )
        context_tables.append(ContextTable(
            table_id=result.table_id,
            title=str(entry["title"]),
            domain=str(entry["domain"]),
            source_path=str(entry["file_path"]),
            retrieval_rank=result.rank,
            retrieval_score=float(result.score),
            columns=columns,
            rows=rows,
        ))
    return tuple(context_tables)


def _render_context(question: str, tables: tuple[ContextTable, ...]) -> str:
    payload = {
        "question": question,
        "tables": [
            {
                "table_id": table.table_id,
                "title": table.title,
                "domain": table.domain,
                "source_path": table.source_path,
                "retrieval_rank": table.retrieval_rank,
                "columns": list(table.columns),
                "rows": [list(row) for row in table.rows],
            }
            for table in tables
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_context(
    question: str,
    retrieval_results: Sequence[RetrievalResult],
    manifest_path: str | Path,
    dataset_dir: str | Path,
    *,
    top_k: int = 3,
    max_characters: int = 100_000,
) -> ContextPackage:
    """Build deterministic context without summarizing or silently dropping rows."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    _positive_int(max_characters, "max_characters")
    question = question.strip()
    tables = load_context_tables(
        retrieval_results, manifest_path, dataset_dir, top_k=top_k,
    )
    context_text = _render_context(question, tables)
    if len(context_text) > max_characters:
        raise ValueError(
            "Complete retrieved tables exceed the context character budget: "
            f"{len(context_text)} > {max_characters}. Increase the budget, lower "
            "top_k, or add an explicit row-selection strategy."
        )
    return ContextPackage(question, tables, context_text)
