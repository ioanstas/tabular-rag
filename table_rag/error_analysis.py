"""Summarize saved per-query retrieval results for analysis and presentation."""
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "query_id", "split", "domain", "query_type", "difficulty", "question",
    "expected_table_ids", "method", "first_relevant_rank",
    "hit_at_1", "hit_at_3", "hit_at_5", "top_5_table_ids",
}


def load_query_results(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
            raise ValueError(f"Query-results CSV is missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Query-results CSV is empty")
    seen = set()
    for row in rows:
        key = (row["query_id"], row["method"])
        if key in seen:
            raise ValueError(f"Duplicate query/method row: {key}")
        seen.add(key)
        if row["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid split for {row['query_id']}")
        try:
            row["first_relevant_rank"] = (
                int(row["first_relevant_rank"]) if row["first_relevant_rank"] else None
            )
            for field in ("hit_at_1", "hit_at_3", "hit_at_5"):
                row[field] = int(row[field])
        except ValueError as exc:
            raise ValueError(f"Invalid rank/hit value for {key}") from exc
        if any(row[field] not in {0, 1} for field in ("hit_at_1", "hit_at_3", "hit_at_5")):
            raise ValueError(f"Hit values must be zero or one for {key}")
    return rows


def failure_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"])].append(row)
    output = []
    for (split, method), group in sorted(groups.items()):
        count = len(group)
        output.append({
            "split": split,
            "method": method,
            "query_count": count,
            "top_1_failures": sum(1 - row["hit_at_1"] for row in group),
            "top_3_failures": sum(1 - row["hit_at_3"] for row in group),
            "top_5_failures": sum(1 - row["hit_at_5"] for row in group),
            "hit_rate_at_1": sum(row["hit_at_1"] for row in group) / count,
            "hit_rate_at_3": sum(row["hit_at_3"] for row in group) / count,
            "hit_rate_at_5": sum(row["hit_at_5"] for row in group) / count,
            "mrr": sum(
                0 if row["first_relevant_rank"] is None
                else 1 / row["first_relevant_rank"]
                for row in group
            ) / count,
        })
    return output


def grouped_metrics(rows: list[dict], field: str) -> list[dict]:
    if field not in {"domain", "query_type"}:
        raise ValueError("Grouping field must be domain or query_type")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"], row[field])].append(row)
    output = []
    for (split, method, value), group in sorted(groups.items()):
        count = len(group)
        output.append({
            "split": split,
            "method": method,
            field: value,
            "query_count": count,
            "hit_rate_at_1": sum(row["hit_at_1"] for row in group) / count,
            "hit_rate_at_3": sum(row["hit_at_3"] for row in group) / count,
            "hit_rate_at_5": sum(row["hit_at_5"] for row in group) / count,
            "mrr": sum(
                0 if row["first_relevant_rank"] is None
                else 1 / row["first_relevant_rank"]
                for row in group
            ) / count,
        })
    return output


def dense_weighted_comparison(
    rows: list[dict], weighted_method: str,
) -> list[dict]:
    by_query = defaultdict(dict)
    for row in rows:
        by_query[(row["split"], row["query_id"])][row["method"]] = row
    output = []
    for (split, query_id), methods in sorted(by_query.items()):
        if "Dense" not in methods or weighted_method not in methods:
            raise ValueError(f"Missing dense/weighted result for {query_id}")
        dense = methods["Dense"]
        weighted = methods[weighted_method]
        dense_rank = dense["first_relevant_rank"]
        weighted_rank = weighted["first_relevant_rank"]
        dense_value = dense_rank if dense_rank is not None else float("inf")
        weighted_value = weighted_rank if weighted_rank is not None else float("inf")
        outcome = (
            "weighted_improved" if weighted_value < dense_value
            else "dense_improved" if dense_value < weighted_value
            else "same_relevant_rank"
        )
        output.append({
            "query_id": query_id,
            "split": split,
            "domain": dense["domain"],
            "query_type": dense["query_type"],
            "question": dense["question"],
            "expected_table_ids": dense["expected_table_ids"],
            "dense_rank": dense_rank,
            "weighted_rank": weighted_rank,
            "outcome": outcome,
            "same_top_5_order": dense["top_5_table_ids"] == weighted["top_5_table_ids"],
            "dense_top_5": dense["top_5_table_ids"],
            "weighted_top_5": weighted["top_5_table_ids"],
        })
    return output
