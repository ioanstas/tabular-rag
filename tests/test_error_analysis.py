"""Test retrieval error summaries without loading models."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from table_rag.error_analysis import (
    dense_weighted_comparison, failure_summary, grouped_metrics, load_query_results,
)


FIELDS = [
    "query_id", "split", "domain", "query_type", "difficulty", "question",
    "expected_table_ids", "method", "first_relevant_rank",
    "hit_at_1", "hit_at_3", "hit_at_5", "top_5_table_ids",
]


def row(query_id, method, rank, *, split="test", query_type="semantic"):
    return {
        "query_id": query_id, "split": split, "domain": "healthcare",
        "query_type": query_type, "difficulty": "medium", "question": "question",
        "expected_table_ids": "t1", "method": method,
        "first_relevant_rank": rank,
        "hit_at_1": int(rank == 1),
        "hit_at_3": int(rank is not None and rank <= 3),
        "hit_at_5": int(rank is not None and rank <= 5),
        "top_5_table_ids": "t1;t2" if rank == 1 else "t2;t1",
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_failure_and_group_summaries(self):
        rows = [row("q1", "Dense", 1), row("q2", "Dense", 4)]
        summary = failure_summary(rows)[0]
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["top_1_failures"], 1)
        self.assertEqual(summary["top_3_failures"], 1)
        self.assertEqual(summary["top_5_failures"], 0)
        self.assertEqual(summary["mrr"], 0.625)
        grouped = grouped_metrics(rows, "query_type")[0]
        self.assertEqual(grouped["hit_rate_at_1"], 0.5)
        with self.assertRaises(ValueError):
            grouped_metrics(rows, "difficulty")

    def test_dense_weighted_comparison(self):
        rows = [
            row("q1", "Dense", 2), row("q1", "Weighted a=0.8", 1),
            row("q2", "Dense", 1), row("q2", "Weighted a=0.8", 2),
            row("q3", "Dense", 3), row("q3", "Weighted a=0.8", 3),
        ]
        result = dense_weighted_comparison(rows, "Weighted a=0.8")
        self.assertEqual([item["outcome"] for item in result],
                         ["weighted_improved", "dense_improved", "same_relevant_rank"])

    def test_csv_validation_and_types(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row("q1", "Dense", 1))
            loaded = load_query_results(path)
            self.assertEqual(loaded[0]["first_relevant_rank"], 1)
            self.assertEqual(loaded[0]["hit_at_1"], 1)
            with path.open("a", encoding="utf-8", newline="") as output:
                csv.DictWriter(output, fieldnames=FIELDS).writerow(row("q1", "Dense", 1))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_query_results(path)


if __name__ == "__main__":
    unittest.main()
