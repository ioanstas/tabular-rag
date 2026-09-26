"""Load retrieval judgments and evaluate ranked table results."""
from dataclasses import dataclass
import csv
from pathlib import Path

from table_rag.sparse_retrieval import RetrievalResult


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    question: str
    split: str
    domain: str
    query_type: str
    difficulty: str
    relevant_table_ids: frozenset[str]
    review_status: str


def load_evaluation_queries(
    queries_path: str | Path, qrels_path: str | Path,
) -> list[EvaluationQuery]:
    """Load answerable questions and their positive table judgments."""
    with Path(qrels_path).open(encoding="utf-8-sig", newline="") as source:
        qrel_rows = list(csv.DictReader(source))
    relevant: dict[str, set[str]] = {}
    qrel_statuses: dict[str, set[str]] = {}
    for row in qrel_rows:
        query_id = row.get("query_id", "").strip()
        table_id = row.get("table_id", "").strip()
        status = row.get("review_status", "").strip()
        try:
            relevance = int(row.get("relevance", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid relevance for {query_id!r}") from exc
        if not query_id or not table_id or not status:
            raise ValueError("Qrels require query_id, table_id, and review_status")
        if relevance > 0:
            relevant.setdefault(query_id, set()).add(table_id)
            qrel_statuses.setdefault(query_id, set()).add(status)

    with Path(queries_path).open(encoding="utf-8-sig", newline="") as source:
        query_rows = list(csv.DictReader(source))
    output = []
    seen = set()
    for row in query_rows:
        query_id = row.get("query_id", "").strip()
        question = row.get("question", "").strip()
        split = row.get("split", "").strip()
        domain = row.get("domain", "").strip()
        query_type = row.get("query_type", "").strip()
        difficulty = row.get("difficulty", "").strip()
        status = row.get("review_status", "").strip()
        answerable = row.get("answerable", "").strip().casefold()
        if not query_id or query_id in seen:
            raise ValueError(f"Missing or duplicate query_id {query_id!r}")
        seen.add(query_id)
        if answerable not in {"true", "false"}:
            raise ValueError(f"Invalid answerable value for {query_id}")
        if (split not in {"dev", "test"} or not question or not domain
                or not query_type or not difficulty or not status):
            raise ValueError(f"Invalid query fields for {query_id}")
        tables = frozenset(relevant.get(query_id, set()))
        if answerable == "true" and not tables:
            raise ValueError(f"Answerable query {query_id} has no positive qrel")
        if answerable == "false" and tables:
            raise ValueError(f"Unanswerable query {query_id} has positive qrels")
        if answerable == "true":
            if qrel_statuses.get(query_id) != {status}:
                raise ValueError(f"Review status mismatch for {query_id}")
            output.append(EvaluationQuery(
                query_id, question, split, domain, query_type, difficulty,
                tables, status,
            ))
    unknown_qrels = set(relevant) - seen
    if unknown_qrels:
        raise ValueError(f"Qrels reference unknown queries: {sorted(unknown_qrels)}")
    return output


def retrieval_metrics(
    queries: list[EvaluationQuery],
    rankings: dict[str, list[RetrievalResult]],
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float | int]:
    """Compute Hit@k and MRR using the first relevant retrieved table."""
    if not queries:
        raise ValueError("Cannot evaluate an empty query set")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in cutoffs):
        raise ValueError("Metric cutoffs must be positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Metric cutoffs must be unique")
    expected = {query.query_id for query in queries}
    if set(rankings) != expected:
        raise ValueError("Rankings must contain exactly the evaluated query IDs")
    hits = {k: 0 for k in cutoffs}
    reciprocal_rank = 0.0
    for query in queries:
        results = rankings[query.query_id]
        ids = [result.table_id for result in results]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate result table for {query.query_id}")
        relevant_ranks = [
            rank for rank, table_id in enumerate(ids, start=1)
            if table_id in query.relevant_table_ids
        ]
        first = min(relevant_ranks, default=None)
        if first is not None:
            reciprocal_rank += 1.0 / first
            for k in cutoffs:
                hits[k] += first <= k
    metrics: dict[str, float | int] = {"query_count": len(queries)}
    metrics.update({f"hit_rate@{k}": hits[k] / len(queries) for k in cutoffs})
    metrics["mrr"] = reciprocal_rank / len(queries)
    return metrics


def select_alpha(rows: list[dict]) -> float:
    """Choose by MRR, then Hit@1/3/5; prefer smaller alpha on exact ties."""
    if not rows:
        raise ValueError("No weighted-fusion results to select from")
    return float(max(
        rows,
        key=lambda row: (
            row["mrr"], row["hit_rate@1"], row["hit_rate@3"],
            row["hit_rate@5"], -row["alpha"],
        ),
    )["alpha"])


def query_result_rows(
    queries: list[EvaluationQuery],
    method_rankings: dict[str, dict[str, list[RetrievalResult]]],
) -> list[dict]:
    """Create presentation-friendly per-query records for fixed methods."""
    if not queries or not method_rankings:
        raise ValueError("Queries and method rankings must be non-empty")
    query_ids = {query.query_id for query in queries}
    if len(query_ids) != len(queries):
        raise ValueError("Evaluation query IDs must be unique")
    for method, rankings in method_rankings.items():
        if not isinstance(method, str) or not method.strip() or set(rankings) != query_ids:
            raise ValueError("Each method must contain exactly the evaluated query IDs")
    rows = []
    for query in queries:
        expected = sorted(query.relevant_table_ids)
        for method, rankings in method_rankings.items():
            results = rankings[query.query_id]
            table_ids = [result.table_id for result in results]
            if len(table_ids) != len(set(table_ids)):
                raise ValueError(f"Duplicate result for {query.query_id}/{method}")
            first_rank = next(
                (rank for rank, table_id in enumerate(table_ids, start=1)
                 if table_id in query.relevant_table_ids),
                None,
            )
            rows.append({
                "query_id": query.query_id,
                "split": query.split,
                "domain": query.domain,
                "query_type": query.query_type,
                "difficulty": query.difficulty,
                "question": query.question,
                "expected_table_ids": ";".join(expected),
                "method": method,
                "first_relevant_rank": first_rank,
                "hit_at_1": int(first_rank is not None and first_rank <= 1),
                "hit_at_3": int(first_rank is not None and first_rank <= 3),
                "hit_at_5": int(first_rank is not None and first_rank <= 5),
                "top_5_table_ids": ";".join(table_ids[:5]),
                "top_5_scores": ";".join(f"{result.score:.8f}" for result in results[:5]),
            })
    return rows
