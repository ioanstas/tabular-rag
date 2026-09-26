"""Test evaluation-data validation, metrics, and development selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from table_rag.retrieval_evaluation import (
    load_evaluation_queries, query_result_rows, retrieval_metrics, select_alpha,
)
from table_rag.sparse_retrieval import RetrievalResult
from run_retrieval_evaluation import _parse_alphas


QUERY_HEADER = (
    "query_id,question,answerable,domain,query_type,difficulty,split,"
    "intended_table_id,source_columns,evidence_note,generation_method,"
    "review_status,reviewer_notes\n"
)
QREL_HEADER = "query_id,table_id,relevance,judgment_basis,review_status\n"


def ranking(*table_ids):
    return [
        RetrievalResult(table_id, rank, 1.0 / rank)
        for rank, table_id in enumerate(table_ids, start=1)
    ]


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.queries_path = root / "queries.csv"
        self.qrels_path = root / "qrels.csv"
        self.queries_path.write_text(
            QUERY_HEADER
            + "q1,first question,true,healthcare,lexical,easy,dev,t1,c,n,g,pending_review,\n"
            + "q2,no answer,false,weather,semantic,hard,dev,,c,n,g,pending_review,\n"
            + "q3,test question,true,football,semantic,medium,test,t3,c,n,g,pending_review,\n",
            encoding="utf-8",
        )
        self.qrels_path.write_text(
            QREL_HEADER
            + "q1,t1,3,direct,pending_review\n"
            + "q1,t2,1,also relevant,pending_review\n"
            + "q3,t3,3,direct,pending_review\n",
            encoding="utf-8",
        )

    def test_loader_keeps_answerable_queries_and_all_positive_qrels(self):
        queries = load_evaluation_queries(self.queries_path, self.qrels_path)
        self.assertEqual([query.query_id for query in queries], ["q1", "q3"])
        self.assertEqual(queries[0].relevant_table_ids, frozenset({"t1", "t2"}))
        self.assertEqual(queries[1].split, "test")

    def test_query_result_rows_preserve_metadata_and_ranks(self):
        queries = load_evaluation_queries(self.queries_path, self.qrels_path)
        rows = query_result_rows(
            queries,
            {"Dense": {"q1": ranking("x", "t2"), "q3": ranking("t3")}},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["query_type"], "lexical")
        self.assertEqual(rows[0]["first_relevant_rank"], 2)
        self.assertEqual(rows[0]["top_5_table_ids"], "x;t2")
        self.assertEqual(rows[1]["hit_at_1"], 1)

    def test_metrics_use_first_relevant_rank(self):
        queries = load_evaluation_queries(self.queries_path, self.qrels_path)
        metrics = retrieval_metrics(
            queries,
            {"q1": ranking("x", "t2"), "q3": ranking("t3", "x")},
        )
        self.assertEqual(metrics["query_count"], 2)
        self.assertEqual(metrics["hit_rate@1"], 0.5)
        self.assertEqual(metrics["hit_rate@3"], 1.0)
        self.assertEqual(metrics["hit_rate@5"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)

    def test_metrics_reject_missing_queries_duplicates_and_bad_cutoffs(self):
        queries = load_evaluation_queries(self.queries_path, self.qrels_path)
        with self.assertRaisesRegex(ValueError, "exactly"):
            retrieval_metrics(queries, {"q1": ranking("t1")})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            retrieval_metrics(
                queries,
                {"q1": ranking("t1", "t1"), "q3": ranking("t3")},
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            retrieval_metrics(queries, {"q1": [], "q3": []}, cutoffs=(0,))

    def test_invalid_ground_truth_is_rejected(self):
        self.qrels_path.write_text(QREL_HEADER, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no positive qrel"):
            load_evaluation_queries(self.queries_path, self.qrels_path)

    def test_alpha_selection_and_parsing(self):
        rows = [
            {"alpha": 0.2, "mrr": 0.8, "hit_rate@1": 0.7,
             "hit_rate@3": 0.9, "hit_rate@5": 1.0},
            {"alpha": 0.4, "mrr": 0.8, "hit_rate@1": 0.7,
             "hit_rate@3": 0.9, "hit_rate@5": 1.0},
        ]
        self.assertEqual(select_alpha(rows), 0.2)
        self.assertEqual(_parse_alphas("0,.4,1"), (0.0, 0.4, 1.0))
        for value in ("", "0.4,0.4", "-.1", "nan", "text"):
            with self.subTest(value=value), self.assertRaises(Exception):
                _parse_alphas(value)


if __name__ == "__main__":
    unittest.main()
