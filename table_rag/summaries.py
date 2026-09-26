"""Validate externally prepared table summaries before embedding them.

This module makes no LLM calls. Text is a versioned input artifact.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from table_rag.ingestion import LoadedTable, load_all_tables, load_manifest

SCHEMA_VERSION = 1
PROMPT_VERSION = "table-summary-v1"
GENERATION_METHOD = "assistant_authored_from_source_profiles"
REVIEW_STATUSES = {"assistant_checked_pending_human_review", "human_approved"}


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_sha256(metadata: dict) -> str:
    # Normalize dimension integers to strings to match manifest CSV values.
    canonical = json.dumps(
        {key: str(value) for key, value in metadata.items()},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return sha256_text(canonical)


def source_profile(table: LoadedTable) -> dict:
    """Compute description facts from every row, without evaluation questions."""
    data = table.data
    time_ranges = {}
    categorical_examples = {}
    # Value patterns, not column substrings: 'candidates' is not a date field.
    period_pattern = re.compile(r"(?:\d{4}-\d{2}(?:-\d{2})?|\d{4}-Q[1-4])")
    number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
    for column in data.columns:
        values = sorted(set(value for value in data[column] if value))
        if values and all(period_pattern.fullmatch(value) for value in values):
            time_ranges[column] = {
                "min": values[0], "max": values[-1], "distinct_count": len(values),
            }
        elif values and not all(number_pattern.fullmatch(value) for value in values):
            categorical_examples[column] = {
                "distinct_count": len(values), "examples": values[:4],
            }
    warnings = []
    declared = [str(table.metadata.get("date_start", "")), str(table.metadata.get("date_end", ""))]
    for column, bounds in time_ranges.items():
        observed = [bounds["min"], bounds["max"]]
        if observed != declared:
            warnings.append({
                "kind": "manifest_date_range_disagrees_with_csv",
                "column": column, "observed": observed, "manifest": declared,
            })
    if "season" in str(table.metadata.get("row_granularity", "")) and "season" not in data.columns:
        warnings.append({"kind": "manifest_grain_mentions_absent_season_column"})
    return {
        "row_count": len(data),
        "columns": data.columns.tolist(),
        "observed_time_ranges": time_ranges,
        "categorical_examples": categorical_examples,
        "missing_counts": {column: int(data[column].eq("").sum()) for column in data.columns},
        "warnings": warnings,
    }


@dataclass(frozen=True)
class TableSummary:
    table_id: str
    summary_text: str
    source_sha256: str
    metadata_sha256: str
    review_status: str


def load_table_summaries(
    path: str | Path, dataset_dir: str | Path, prompt_path: str | Path,
) -> list[TableSummary]:
    """Require one current summary per manifest table; retain manifest order.

    Integrity and source-fact checks do not prove every prose claim true.
    Pending human review is permitted for this explicitly labelled baseline.
    Changed sources or instructions require a new summary release.
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir / "table_manifest.csv")
    tables = load_all_tables(manifest, dataset_dir)
    expected_prompt_hash = sha256_file(prompt_path)
    summaries = {}
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Summary line {number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Summary line {number}: expected an object")
            table_id = record.get("table_id")
            if not isinstance(table_id, str) or table_id not in tables:
                raise ValueError(f"Unknown summary table_id on line {number}: {table_id!r}")
            if table_id in summaries:
                raise ValueError(f"Duplicate summary: {table_id}")
            text = record.get("summary_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty summary: {table_id}")
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Incompatible summary schema: {table_id}")
            if (record.get("prompt_version") != PROMPT_VERSION
                    or record.get("prompt_sha256") != expected_prompt_hash):
                raise ValueError(f"Summary instructions changed: {table_id}; prepare a new release")
            if record.get("generation_method") != GENERATION_METHOD:
                raise ValueError(f"Unexpected summary generation method: {table_id}")
            if record.get("review_status") not in REVIEW_STATUSES:
                raise ValueError(f"Invalid summary review status: {table_id}")
            if record.get("summary_sha256") != sha256_text(text):
                raise ValueError(f"Summary text checksum mismatch: {table_id}")
            table = tables[table_id]
            actual_hash = sha256_file(dataset_dir / str(table.metadata["file_path"]))
            if record.get("source_sha256") != actual_hash:
                raise ValueError(f"Stale summary source: {table_id}; prepare a new release")
            if record.get("metadata_sha256") != metadata_sha256(table.metadata):
                raise ValueError(f"Stale summary metadata: {table_id}; prepare a new release")
            if record.get("source_profile") != source_profile(table):
                raise ValueError(f"Summary source facts disagree with CSV: {table_id}")
            summaries[table_id] = TableSummary(
                table_id, text, actual_hash, record["metadata_sha256"], record["review_status"],
            )
    missing = set(tables) - set(summaries)
    if missing:
        raise ValueError(f"Missing summaries: {sorted(missing)}")
    return [summaries[table_id] for table_id in tables]
