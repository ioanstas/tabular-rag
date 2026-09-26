"""Load and validate the catalogue of source tables."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Read metadata without loading source tables.

    Preserve text and row order; convert dimensions to integers.
    Raise ValueError for invalid contents; propagate filesystem errors.
    """
    manifest_path = Path(manifest_path)
    required = ("table_id", "file_path", "title", "domain", "row_count", "column_count")
    try:
        manifest = pd.read_csv(
            manifest_path, encoding="utf-8", dtype="string", keep_default_na=False
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Manifest is empty: {manifest_path}") from exc
    missing = [name for name in required if name not in manifest.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")
    if manifest.empty:
        raise ValueError(f"Manifest has no entries: {manifest_path}")
    for name in required:
        blank = manifest[name].str.strip().eq("")
        if blank.any():
            rows = (manifest.index[blank] + 1).tolist()
            raise ValueError(f"Blank '{name}' in manifest entries {rows}")
    duplicates = manifest.loc[
        manifest["table_id"].duplicated(keep=False), "table_id"
    ].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate table_id values: {duplicates}")
    for name in ("row_count", "column_count"):
        values = manifest[name].str.strip()
        invalid = ~values.str.fullmatch(r"[0-9]+")
        if invalid.any():
            ids = manifest.loc[invalid, "table_id"].tolist()
            raise ValueError(
                f"'{name}' must contain non-negative integers; invalid table IDs: {ids}"
            )
        manifest[name] = values.map(int)
    return manifest


@dataclass
class LoadedTable:
    """A source table with its identity and manifest metadata attached."""

    table_id: str
    data: pd.DataFrame
    metadata: dict[str, str | int]


def load_table(entry: pd.Series, dataset_dir: str | Path) -> LoadedTable:
    """Load one validated manifest entry and check its expected dimensions.

    Cell values stay as text, including numeric formatting and empty cells.
    Paths in manifest entries are relative to dataset_dir.
    """
    table_id = str(entry["table_id"])
    source_path = (Path(dataset_dir) / entry["file_path"]).resolve()
    context = f"Table '{table_id}' at {source_path}"
    try:
        data = pd.read_csv(
            source_path, encoding="utf-8", dtype="string", keep_default_na=False
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{context}: source file not found") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f"{context}: CSV could not be read: {exc}") from exc
    except OSError as exc:
        raise OSError(f"{context}: source file could not be read: {exc}") from exc

    expected_shape = (int(entry["row_count"]), int(entry["column_count"]))
    if data.shape != expected_shape:
        raise ValueError(
            f"{context}: dimension mismatch; expected {expected_shape}, "
            f"actual {data.shape}"
        )

    return LoadedTable(table_id=table_id, data=data, metadata=entry.to_dict())


def load_all_tables(
    manifest: pd.DataFrame, dataset_dir: str | Path
) -> dict[str, LoadedTable]:
    """Load a validated manifest sequentially, stopping on the first error.

    The returned dictionary preserves manifest order. Duplicate IDs are
    rejected before any source file is loaded.
    """
    duplicates = manifest.loc[
        manifest["table_id"].duplicated(keep=False), "table_id"
    ].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate table_id values: {duplicates}")

    tables = {}
    for _, entry in manifest.iterrows():
        table = load_table(entry, dataset_dir)
        tables[table.table_id] = table
    return tables
