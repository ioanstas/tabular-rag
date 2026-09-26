"""Verify fusion arithmetic, boundaries, and shared persisted-index retrieval."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from table_rag.dense_index import DIMENSION, MAX_SEQUENCE_LENGTH, build_dense_index, save_dense_index
from table_rag.hybrid_retrieval import fuse_rrf, fuse_weighted, retrieve_candidates
from table_rag.sparse_retrieval import (
    RetrievalResult, SparseDocument, build_bm25, bm25_metadata, save_bm25,
)
from table_rag.summaries import TableSummary, sha256_file
import run_hybrid_retrieval


def ranked(*items):
    return [RetrievalResult(table_id, rank, score)
            for rank, (table_id, score) in enumerate(items, start=1)]


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.sparse = ranked(("a", 10), ("b", 6), ("c", 2))
        self.dense = ranked(("b", 0.9), ("a", 0.5), ("d", 0.1))

    def test_rrf_formula_union_ties_and_ranks(self):
        results = fuse_rrf(self.sparse, self.dense, 10)
        self.assertEqual([r.table_id for r in results], ["a", "b", "c", "d"])
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])
        self.assertAlmostEqual(results[0].score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(results[2].score, 1 / 63)
        self.assertAlmostEqual(fuse_rrf(self.sparse, [], 1, rank_constant=1)[0].score, 0.5)

    def test_rrf_ignores_score_magnitudes(self):
        changed = ranked(("a", 1000), ("b", -2), ("c", -900))
        self.assertEqual(fuse_rrf(changed, self.dense), fuse_rrf(self.sparse, self.dense))

    def test_weighted_arithmetic_and_alpha_changes_winner(self):
        results = fuse_weighted(self.sparse, self.dense, 10, alpha=0.4)
        self.assertEqual([r.table_id for r in results], ["a", "b", "c", "d"])
        self.assertAlmostEqual(results[0].score, 0.8)
        self.assertAlmostEqual(results[1].score, 0.7)
        self.assertEqual(results[2].score, 0)
        self.assertEqual(fuse_weighted(self.sparse, self.dense, 1, alpha=0.8)[0].table_id, "b")

    def test_alpha_endpoints_exclude_inactive_candidates(self):
        for alpha, expected in ((0, ["a", "b", "c"]), (1, ["b", "a", "d"])):
            results = fuse_weighted(self.sparse, self.dense, 10, alpha=alpha)
            self.assertEqual([r.table_id for r in results], expected)
        self.assertEqual(fuse_weighted([], self.dense, alpha=0), [])
        self.assertEqual(fuse_weighted(self.sparse, [], alpha=1), [])

    def test_empty_singleton_constant_and_negative_scores(self):
        self.assertEqual(fuse_rrf([], []), [])
        self.assertEqual(fuse_weighted([], []), [])
        singleton = ranked(("x", -3))
        self.assertAlmostEqual(fuse_weighted(singleton, [], alpha=0.4)[0].score, 0.6)
        constant = ranked(("b", 0), ("a", 0))
        result = fuse_weighted(constant, [], alpha=0)
        self.assertEqual([r.table_id for r in result], ["a", "b"])
        self.assertEqual([r.score for r in result], [1, 1])
        negative = ranked(("a", -1), ("b", -2), ("c", -3))
        self.assertEqual([r.score for r in fuse_weighted(negative, [], alpha=0)], [1, 0.5, 0])
        self.assertEqual([r.table_id for r in fuse_rrf([], self.dense)], ["b", "a", "d"])

    def test_invalid_settings_and_malformed_rankings(self):
        for bad in (0, -1, True, 1.5):
            for fuse in (fuse_rrf, fuse_weighted):
                with self.subTest(top_k=bad, fuse=fuse), self.assertRaises(ValueError):
                    fuse([], [], top_k=bad)
            with self.assertRaises(ValueError):
                fuse_rrf([], [], rank_constant=bad)
        for alpha in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                fuse_weighted([], [], alpha=alpha)
        malformed = [
            ranked(("a", 2), ("a", 1)), ranked(("a", 1), ("b", 2)),
            ranked(("a", float("nan"))), ranked(("a", float("inf"))),
            [RetrievalResult("a", 2, 1)], ranked(("", 1)),
        ]
        for results in malformed:
            for fuse in (fuse_rrf, fuse_weighted):
                with self.subTest(results=results, fuse=fuse), self.assertRaises(ValueError):
                    fuse(results, [])

    def test_retrieve_full_rankings_once_and_match_ids_not_positions(self):
        sparse_index = SimpleNamespace(table_ids=("a", "b", "c"))
        dense_index = SimpleNamespace(table_ids=("c", "b", "a"))
        model = object()
        with patch("table_rag.hybrid_retrieval.search_bm25", return_value=self.sparse) as sparse, \
             patch("table_rag.hybrid_retrieval.search_dense", return_value=self.dense) as dense:
            results = retrieve_candidates(sparse_index, dense_index, model, "question")
            self.assertEqual(results, (self.sparse, self.dense))
            sparse.assert_called_once_with(sparse_index, "question", top_k=3)
            dense.assert_called_once_with(dense_index, model, "question", top_k=3)
            sparse.reset_mock()
            dense.reset_mock()
            for question in ("", " ", None):
                with self.assertRaises(ValueError):
                    retrieve_candidates(sparse_index, dense_index, model, question)
            with self.assertRaisesRegex(ValueError, "same table IDs"):
                retrieve_candidates(sparse_index, SimpleNamespace(table_ids=("x",)), model, "question")
            sparse.assert_not_called()
            dense.assert_not_called()


class HybridRunnerTests(unittest.TestCase):
    def test_saved_indexes_both_methods_encode_once_and_stale_source_fails(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = artifacts / "sparse_documents.jsonl"
            source.write_text("fixture documents", encoding="utf-8")
            summary = artifacts / "table_summaries.jsonl"
            summary.write_text("fixture summaries", encoding="utf-8")
            sparse = build_bm25([
                SparseDocument("a", "hospital admissions"),
                SparseDocument("b", "hotel bookings"),
                SparseDocument("c", "football goals"),
            ])
            sparse_path = artifacts / "bm25_index.pkl"
            save_bm25(sparse, bm25_metadata(source), sparse_path)
            model = Mock()
            model.max_seq_length = MAX_SEQUENCE_LENGTH
            model.tokenizer.side_effect = lambda texts, **kwargs: {
                "input_ids": [[1, 2, 3] for _ in texts]
            }
            model.encode.return_value = np.eye(3, DIMENSION, dtype=np.float32)
            summaries = [TableSummary(i, "summary", "a" * 64, "b" * 64, "human_approved")
                         for i in ("a", "b", "c")]
            dense = build_dense_index(summaries, sha256_file(summary), model)
            dense_path = artifacts / "dense_index.npz"
            save_dense_index(dense, dense_path)
            before = (sparse_path.read_bytes(), dense_path.read_bytes())
            model.reset_mock()
            model.encode.return_value = np.eye(1, DIMENSION, dtype=np.float32)
            with patch.object(run_hybrid_retrieval, "__file__", str(root / "run_hybrid_retrieval.py")), \
                 patch("sys.argv", ["run_hybrid_retrieval.py", "hospital", "--method", "both",
                                    "--alpha", "0.4", "--local-files-only"]), \
                 patch.object(run_hybrid_retrieval, "load_embedding_model", return_value=model) as loader:
                output = io.StringIO()
                with redirect_stdout(output):
                    run_hybrid_retrieval.main()
                self.assertIn("RRF (", output.getvalue())
                self.assertIn("Weighted (alpha=0.4", output.getvalue())
                self.assertEqual(output.getvalue().count("1. a "), 2)
                loader.assert_called_once_with(local_files_only=True)
                model.encode.assert_called_once()
                self.assertEqual(model.encode.call_args.args[0], ["hospital"])
                self.assertEqual(before, (sparse_path.read_bytes(), dense_path.read_bytes()))
                # Check freshness of each index before the costly model load.
                for path in (summary, source):
                    old = path.read_text(encoding="utf-8")
                    path.write_text("changed", encoding="utf-8")
                    loader.reset_mock()
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        run_hybrid_retrieval.main()
                    self.assertEqual(caught.exception.code, 1)
                    loader.assert_not_called()
                    path.write_text(old, encoding="utf-8")

    def test_cli_rejects_invalid_arguments_before_loading(self):
        for args in (["--alpha", "nan"], ["--top-k", "0"], ["--rank-constant", "-1"],
                     ["--method", "unknown"]):
            with patch("sys.argv", ["run_hybrid_retrieval.py", "question"] + args), \
                 patch.object(run_hybrid_retrieval, "load_bm25") as loader, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                run_hybrid_retrieval.main()
            self.assertEqual(caught.exception.code, 2)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
