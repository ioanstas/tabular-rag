"""Combine existing sparse and dense rankings; no new index is required."""
import math

from table_rag.dense_index import DenseIndex
from table_rag.dense_retrieval import search_dense
from table_rag.sparse_retrieval import RetrievalResult, SparseIndex, search_bm25


def _positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_results(results: list[RetrievalResult]) -> None:
    """Fusion expects complete, ordered rankings from the existing retrievers."""
    seen = set()
    previous_score = math.inf
    for rank, result in enumerate(results, start=1):
        if not isinstance(result.table_id, str) or not result.table_id.strip():
            raise ValueError("Results must have non-empty table IDs")
        if result.table_id in seen:
            raise ValueError("Duplicate table ID in a retrieval list")
        if isinstance(result.rank, bool) or not isinstance(result.rank, int) or result.rank != rank:
            raise ValueError("Results must have consecutive ranks starting at 1")
        if not math.isfinite(result.score) or result.score > previous_score:
            raise ValueError("Results must have finite scores in descending order")
        seen.add(result.table_id)
        previous_score = result.score


def _rank(scores: dict[str, float], top_k: int) -> list[RetrievalResult]:
    ordered = sorted(scores, key=lambda table_id: (-scores[table_id], table_id))
    return [
        RetrievalResult(table_id, rank, scores[table_id])
        for rank, table_id in enumerate(ordered[:top_k], start=1)
    ]


def fuse_rrf(
    sparse: list[RetrievalResult], dense: list[RetrievalResult],
    top_k: int = 5, *, rank_constant: int = 60,
) -> list[RetrievalResult]:
    """Sum 1/(rank_constant + rank) per list; absent tables contribute zero."""
    _positive_int(top_k, "top_k")
    _positive_int(rank_constant, "rank_constant")
    scores = {}
    for results in (sparse, dense):
        _validate_results(results)
        for result in results:
            scores[result.table_id] = (
                scores.get(result.table_id, 0.0) + 1.0 / (rank_constant + result.rank)
            )
    return _rank(scores, top_k)


def _normalize(results: list[RetrievalResult]) -> dict[str, float]:
    """Per-query min-max scores. Empty -> {}; singleton/all tied -> all 1.

    Missing candidates receive zero in fusion. A constant list cannot
    distinguish its members, so each present member receives equal support.
    These normalized scores are not probabilities.
    """
    if not results:
        return {}
    low = min(result.score for result in results)
    high = max(result.score for result in results)
    if high == low:
        return {result.table_id: 1.0 for result in results}
    return {result.table_id: (result.score - low) / (high - low) for result in results}


def fuse_weighted(
    sparse: list[RetrievalResult], dense: list[RetrievalResult],
    top_k: int = 5, *, alpha: float = 0.5,
) -> list[RetrievalResult]:
    """Combine min-max scores: alpha*dense + (1-alpha)*sparse.

    alpha=0 returns sparse candidates only; alpha=1 returns dense only.
    No automatic weight redistribution occurs when one list is empty.
    """
    _positive_int(top_k, "top_k")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must be a finite number between 0 and 1")
    _validate_results(sparse)
    _validate_results(dense)
    sparse_scores = _normalize(sparse) if alpha < 1 else {}
    dense_scores = _normalize(dense) if alpha > 0 else {}
    scores = {
        table_id: (1 - alpha) * sparse_scores.get(table_id, 0.0)
                  + alpha * dense_scores.get(table_id, 0.0)
        for table_id in sparse_scores.keys() | dense_scores.keys()
    }
    return _rank(scores, top_k)


def retrieve_candidates(
    sparse_index: SparseIndex, dense_index: DenseIndex, model, question: str,
) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
    """Retrieve full rankings once, using the pinned model from load_embedding_model().

    Full rankings are inexpensive for our 60 tables and avoid cutting off
    useful candidates before fusion. BM25 returns only lexical matches.
    Match by table ID; index row orders do not need to be identical.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if set(sparse_index.table_ids) != set(dense_index.table_ids):
        raise ValueError("Sparse and dense indexes must cover the same table IDs")
    count = len(dense_index.table_ids)
    sparse = search_bm25(sparse_index, question, top_k=count)
    dense = search_dense(dense_index, model, question, top_k=count)
    return sparse, dense
