"""Load, tokenize, index, and search serialized table documents."""
from dataclasses import dataclass
import json
from pathlib import Path
import re
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class SparseDocument:
    table_id: str
    serialized_text: str


@dataclass(frozen=True)
class RetrievalResult:
    table_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class SparseIndex:
    # The same position identifies a table ID and a BM25 document.
    table_ids: tuple[str, ...]
    model: BM25Okapi


def load_sparse_documents(path: str | Path) -> list[SparseDocument]:
    """Read JSONL in file order and reject invalid records or duplicate IDs."""
    documents = []
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {number}: expected a JSON object")
            for field in ("table_id", "serialized_text"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Line {number}: {field} must be non-empty text")
            table_id = record["table_id"]
            if table_id in seen:
                raise ValueError(f"Line {number}: duplicate table_id {table_id!r}")
            seen.add(table_id)
            documents.append(SparseDocument(table_id, record["serialized_text"]))
    if not documents:
        raise ValueError("No sparse documents found")
    return documents


def tokenize(text: str) -> list[str]:
    """Casefold words; split underscores; keep internal dots and hyphens.

    hospital_name becomes hospital, name; 5.40 and 2025-01 stay intact.
    No stemming or stopword removal. Hyphenated terms require the same
    spelling. Original serialized text is not modified.
    """
    return re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", text.casefold())


def build_bm25(documents: list[SparseDocument]) -> SparseIndex:
    """Compute collection statistics once, preserving the ID mapping."""
    if not documents:
        raise ValueError("Cannot build BM25 without documents")
    table_ids = tuple(document.table_id for document in documents)
    if len(set(table_ids)) != len(table_ids):
        raise ValueError("Duplicate table IDs in BM25 input")
    tokenized_documents = []
    for document in documents:
        tokens = tokenize(document.serialized_text)
        if not tokens:
            raise ValueError(f"No searchable terms in {document.table_id!r}")
        tokenized_documents.append(tokens)
    model = BM25Okapi(tokenized_documents, **BM25_SETTINGS)
    return SparseIndex(table_ids, model)


def search_bm25(index: SparseIndex, question: str, top_k: int = 5) -> list[RetrievalResult]:
    """Rank lexical matches by score, breaking ties by table ID.

    Overlap is checked independently because BM25Okapi scores can be zero
    or negative in small corpora. Matching terms do not prove answerability.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    tokens = tokenize(question)
    if not tokens:
        return []
    terms = set(tokens)
    positions = [
        i for i, frequencies in enumerate(index.model.doc_freqs)
        if not terms.isdisjoint(frequencies)
    ]
    if not positions:
        return []
    scores = index.model.get_scores(tokens)
    positions.sort(key=lambda i: (-float(scores[i]), index.table_ids[i]))
    return [
        RetrievalResult(index.table_ids[i], rank, float(scores[i]))
        for rank, i in enumerate(positions[:top_k], start=1)
    ]

# Bump TOKENIZER_VERSION whenever tokenize() changes.
ARTIFACT_VERSION = 1
TOKENIZER_VERSION = "v1"
BM25_SETTINGS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


def source_sha256(path: str | Path) -> str:
    """Identify the exact serialized documents used for indexing."""
    import hashlib
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bm25_metadata(source_path: str | Path) -> dict:
    """Record settings and dependencies needed to reproduce this index."""
    from importlib.metadata import version
    return {
        "artifact_version": ARTIFACT_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "rank_bm25_version": version("rank-bm25"),
        "bm25_settings": dict(BM25_SETTINGS),
        "source_sha256": source_sha256(source_path),
    }


def _validate_package(package: object) -> SparseIndex:
    """Validate a trusted local artifact before using it."""
    from importlib.metadata import version
    rebuild = "Run uv run python run_sparse_indexing.py to rebuild."
    if not isinstance(package, dict):
        raise ValueError(f"Invalid BM25 artifact. {rebuild}")
    metadata = package.get("metadata")
    index = package.get("index")
    if not isinstance(metadata, dict) or not isinstance(index, SparseIndex):
        raise ValueError(f"Missing BM25 index or metadata. {rebuild}")
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "rank_bm25_version": version("rank-bm25"),
        "bm25_settings": BM25_SETTINGS,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Incompatible BM25 {field}. {rebuild}")
    digest = metadata.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Invalid BM25 source hash. {rebuild}")
    ids, model = index.table_ids, index.model
    if (not isinstance(ids, tuple) or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(model, BM25Okapi)):
        raise ValueError(f"Invalid BM25 table mapping or model. {rebuild}")
    if (len(ids) != model.corpus_size
            or len(ids) != len(model.doc_freqs)
            or len(ids) != len(model.doc_len)):
        raise ValueError(f"BM25 table-ID count does not match model. {rebuild}")
    if any(getattr(model, name) != value for name, value in BM25_SETTINGS.items()):
        raise ValueError(f"BM25 model settings disagree with metadata. {rebuild}")
    return index


def save_bm25(index: SparseIndex, metadata: dict, path: str | Path) -> None:
    """Atomically save the model, ID mapping, and metadata in a local pickle."""
    import os
    import pickle
    from tempfile import NamedTemporaryFile
    package = {"index": index, "metadata": metadata}
    _validate_package(package)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            pickle.dump(package, output, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_bm25(path: str | Path, source_path: str | Path | None = None) -> SparseIndex:
    """Load only trusted, locally generated pickle files.

    Pickle can execute code before validation. Optional source_path checks
    freshness without tokenizing or rebuilding the corpus.
    """
    import pickle
    rebuild = "Run uv run python run_sparse_indexing.py to rebuild."
    try:
        with Path(path).open("rb") as source:
            package = pickle.load(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"BM25 index not found: {path}. {rebuild}") from exc
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError,
            TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read BM25 artifact. {rebuild}") from exc
    index = _validate_package(package)
    if source_path is not None:
        if source_sha256(source_path) != package["metadata"]["source_sha256"]:
            raise ValueError(f"BM25 index is stale: source documents changed. {rebuild}")
    return index
