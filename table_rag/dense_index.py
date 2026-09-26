"""Build and persist the offline dense index; no query retrieval here."""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import zipfile

import numpy as np

from table_rag.summaries import TableSummary

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256
ARTIFACT_VERSION = 1
ENCODING = {
    "normalize_embeddings": True,
    "precision": "float32",
    "prompt": "",
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
}
REBUILD_MESSAGE = "Run uv run python run_dense_indexing.py --rebuild."


@dataclass(frozen=True)
class DenseIndex:
    """vectors[i] belongs to table_ids[i]; metadata identifies the encoder/input."""
    table_ids: tuple[str, ...]
    vectors: np.ndarray
    metadata: dict


def load_embedding_model(*, local_files_only: bool = False):
    """Load the pinned CPU model; download public weights only if needed."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cpu",
        local_files_only=local_files_only, trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    if model.get_embedding_dimension() != DIMENSION:
        raise ValueError("Embedding model returned an unexpected dimension")
    if model.max_seq_length != MAX_SEQUENCE_LENGTH:
        raise ValueError("Embedding model returned an unexpected token limit")
    return model


def token_lengths(model, texts: list[str]) -> list[int]:
    """Count the model's tokens, including special tokens, without truncating."""
    encoded = model.tokenizer(
        texts, add_special_tokens=True, padding=False, truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def embed_texts(model, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    """Encode original text; reject blank or oversized inputs before inference."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty strings")
    if model.max_seq_length != MAX_SEQUENCE_LENGTH:
        raise ValueError("Model token limit differs from the configured encoder")
    lengths = token_lengths(model, texts)
    oversized = [(i, size) for i, size in enumerate(lengths) if size > MAX_SEQUENCE_LENGTH]
    if oversized:
        raise ValueError(f"Inputs exceed {MAX_SEQUENCE_LENGTH} tokens (position, length): {oversized}")
    vectors = np.asarray(model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        precision="float32", convert_to_numpy=True, show_progress_bar=False,
        prompt="",
    ), dtype=np.float32)
    _validate_vectors(vectors, len(texts))
    return vectors


def _validate_vectors(vectors: np.ndarray, count: int) -> None:
    if (not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
            or vectors.shape != (count, DIMENSION) or count < 1):
        raise ValueError(f"Dense vectors must have shape ({count}, {DIMENSION}) and dtype float32")
    if not np.isfinite(vectors).all():
        raise ValueError("Dense vectors contain non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Dense vectors must have unit length")


def _embedding_versions() -> dict[str, str]:
    return {package: version(package) for package in ("sentence-transformers", "transformers", "torch")}


def _payload_sha256(table_ids: tuple[str, ...], vectors: np.ndarray) -> str:
    """Bind vector bytes to the ordered IDs, detecting accidental corruption."""
    digest = hashlib.sha256(json.dumps(table_ids, ensure_ascii=False).encode("utf-8"))
    digest.update(np.ascontiguousarray(vectors, dtype="<f4").tobytes())
    return digest.hexdigest()


def build_dense_index(
    summaries: list[TableSummary], summary_file_sha256: str, model, *, batch_size: int = 16,
) -> DenseIndex:
    """Encode validated summaries using a model from load_embedding_model()."""
    if not summaries:
        raise ValueError("Cannot build a dense index without summaries")
    if re.fullmatch(r"[0-9a-f]{64}", summary_file_sha256) is None:
        raise ValueError("Invalid summary-file SHA-256")
    table_ids = tuple(summary.table_id for summary in summaries)
    if (any(not isinstance(value, str) or not value.strip() for value in table_ids)
            or len(set(table_ids)) != len(table_ids)):
        raise ValueError("Dense table IDs must be unique non-empty strings")
    texts = [summary.summary_text for summary in summaries]
    vectors = embed_texts(model, texts, batch_size=batch_size)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "encoding": dict(ENCODING),
        "summary_file_sha256": summary_file_sha256,
        "embedding_library_versions": _embedding_versions(),
        "numpy_version": version("numpy"),
        "device": "cpu",
        "batch_size": batch_size,
        "summary_review_status_counts": {
            status: sum(summary.review_status == status for summary in summaries)
            for status in sorted({summary.review_status for summary in summaries})
        },
        "max_observed_tokens": max(token_lengths(model, texts)),
        "payload_sha256": _payload_sha256(table_ids, vectors),
    }
    return DenseIndex(table_ids, vectors, metadata)


def validate_dense_index(index: DenseIndex, *, summary_file_sha256: str | None = None) -> None:
    """Validate identity, vectors, encoder settings, and optional input freshness."""
    ids = index.table_ids
    if (not isinstance(ids, tuple) or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid dense table-ID mapping")
    _validate_vectors(index.vectors, len(ids))
    metadata = index.metadata
    if not isinstance(metadata, dict):
        raise ValueError("Dense metadata must be an object")
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "encoding": ENCODING,
        "embedding_library_versions": _embedding_versions(),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Incompatible dense {field}. {REBUILD_MESSAGE}")
    source_hash = metadata.get("summary_file_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError("Invalid dense summary-file hash")
    if summary_file_sha256 is not None and source_hash != summary_file_sha256:
        raise ValueError(f"Dense index is stale: summaries changed. {REBUILD_MESSAGE}")
    if metadata.get("payload_sha256") != _payload_sha256(ids, index.vectors):
        raise ValueError(f"Dense vector/ID checksum mismatch. {REBUILD_MESSAGE}")


def save_dense_index(index: DenseIndex, path: str | Path) -> None:
    """Atomically save vectors, IDs, and JSON metadata together, without pickle."""
    validate_dense_index(index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            np.savez_compressed(
                output, vectors=index.vectors,
                table_ids=np.asarray(index.table_ids, dtype=np.str_),
                metadata=np.asarray(json.dumps(index.metadata, sort_keys=True)),
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_dense_index(
    path: str | Path, *, summary_file_sha256: str | None = None,
) -> DenseIndex:
    """Load only saved arrays/metadata; do not load a model or call inference."""
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"vectors", "table_ids", "metadata"}:
                raise ValueError("Unexpected dense artifact fields")
            ids = artifact["table_ids"]
            if ids.ndim != 1 or ids.dtype.kind != "U":
                raise ValueError("Invalid dense table-ID array")
            metadata = artifact["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("Invalid dense metadata array")
            index = DenseIndex(tuple(ids.tolist()), artifact["vectors"], json.loads(metadata.item()))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dense index not found: {path}. {REBUILD_MESSAGE}") from exc
    except (ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot read dense index: {exc}. {REBUILD_MESSAGE}") from exc
    validate_dense_index(index, summary_file_sha256=summary_file_sha256)
    return index
