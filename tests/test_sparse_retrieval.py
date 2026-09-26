"""Focused checks for loading, identity mapping, and ranking edge cases."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from table_rag.sparse_retrieval import (
    SparseDocument, build_bm25, load_sparse_documents, search_bm25, tokenize,
)


class SparseRetrievalTests(unittest.TestCase):
    def test_loading_and_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "docs.jsonl"
            records = [{"table_id": "b", "serialized_text": "hospital"},
                       {"table_id": "a", "serialized_text": "hotel"}]
            path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            self.assertEqual([d.table_id for d in load_sparse_documents(path)], ["b", "a"])
            for invalid in ("", "bad json", "[]", "{}",
                            json.dumps({"table_id": "a", "serialized_text": " "}),
                            "\n".join(map(json.dumps, [records[0], records[0]]))):
                with self.subTest(invalid=invalid):
                    path.write_text(invalid, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_sparse_documents(path)

    def test_tokenization(self):
        self.assertEqual(tokenize('Hospital_name: "2025-01", 5.40 AB-12'),
                         ["hospital", "name", "2025-01", "5.40", "ab-12"])

    def test_ranking_identity_and_no_matches(self):
        index = build_bm25([SparseDocument("hotel", "hotel bookings"),
                            SparseDocument("hospital", "hospital admissions"),
                            SparseDocument("school", "school enrollment")])
        results = search_bm25(index, "HOSPITAL admissions", 10)
        self.assertEqual([r.table_id for r in results], ["hospital"])
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].score, 0)
        self.assertEqual(search_bm25(index, "unseenword"), [])
        self.assertEqual(search_bm25(index, "?!"), [])

    def test_zero_score_matches_and_ties(self):
        index = build_bm25([SparseDocument("b", "hotel"), SparseDocument("a", "hotel"),
                            SparseDocument("d", "school"), SparseDocument("c", "school")])
        results = search_bm25(index, "hotel")
        self.assertEqual([r.table_id for r in results], ["a", "b"])
        self.assertEqual([r.score for r in results], [0.0, 0.0])
        self.assertEqual(len(search_bm25(index, "hotel", 1)), 1)

    def test_negative_score_match(self):
        index = build_bm25([SparseDocument("a", "hotel")])
        self.assertEqual(search_bm25(index, "hotel")[0].table_id, "a")

    def test_invalid_inputs(self):
        for documents in ([], [SparseDocument("a", "!!!")],
                          [SparseDocument("a", "hotel")] * 2):
            with self.assertRaises(ValueError):
                build_bm25(documents)
        index = build_bm25([SparseDocument("a", "hotel")])
        for top_k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                search_bm25(index, "hotel", top_k)


if __name__ == "__main__":
    unittest.main()
