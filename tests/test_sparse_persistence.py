"""Persistence checks, including search without index reconstruction."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from table_rag.sparse_retrieval import (
    SparseIndex, bm25_metadata, build_bm25, load_bm25,
    load_sparse_documents, save_bm25, search_bm25,
)
import run_sparse_indexing
import run_sparse_retrieval


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "artifacts" / "sparse_documents.jsonl"
        self.source.parent.mkdir()
        records = [
            {"table_id": "b", "serialized_text": "hospital admissions"},
            {"table_id": "a", "serialized_text": "hotel bookings"},
            {"table_id": "c", "serialized_text": "school enrollment"},
        ]
        self.source.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        self.index = build_bm25(load_sparse_documents(self.source))
        self.metadata = bm25_metadata(self.source)
        self.path = self.source.parent / "bm25_index.pkl"

    def test_round_trip_preserves_scores_and_mapping_without_constructor(self):
        save_bm25(self.index, self.metadata, self.path)
        with patch("rank_bm25.BM25Okapi.__init__", side_effect=AssertionError("Rebuilt!")):
            loaded = load_bm25(self.path, self.source)
            self.assertEqual(loaded.table_ids, self.index.table_ids)
            for query in ("hospital admissions", "HOTEL", "unseen", ""):
                self.assertEqual(search_bm25(loaded, query), search_bm25(self.index, query))

    def test_metadata_is_saved_and_source_changes_are_detected(self):
        save_bm25(self.index, self.metadata, self.path)
        with self.path.open("rb") as source:
            self.assertEqual(pickle.load(source)["metadata"], self.metadata)
        self.source.write_text(self.source.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            load_bm25(self.path, self.source)

    def test_versions_settings_hash_and_mapping_are_validated(self):
        cases = []
        for field in ("artifact_version", "tokenizer_version", "rank_bm25_version",
                      "bm25_settings", "source_sha256"):
            metadata = dict(self.metadata)
            metadata[field] = "wrong"
            cases.append({"index": self.index, "metadata": metadata})
        cases.extend([
            {"index": SparseIndex(("a",), self.index.model), "metadata": self.metadata},
            {"index": SparseIndex(("a", "a", "c"), self.index.model), "metadata": self.metadata},
            {"index": self.index},
            [],
        ])
        model = copy.deepcopy(self.index.model)
        model.k1 = 99
        cases.append({"index": SparseIndex(self.index.table_ids, model), "metadata": self.metadata})
        for package in cases:
            with self.subTest(package_type=type(package)):
                self.path.write_bytes(pickle.dumps(package))
                with self.assertRaisesRegex(ValueError, "run_sparse_indexing.py"):
                    load_bm25(self.path)

    def test_missing_and_corrupt_artifacts(self):
        with self.assertRaisesRegex(FileNotFoundError, "run_sparse_indexing.py"):
            load_bm25(self.path)
        for data in (b"", b"not a pickle"):
            self.path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "run_sparse_indexing.py"):
                load_bm25(self.path)

    def test_failed_save_preserves_previous_artifact(self):
        save_bm25(self.index, self.metadata, self.path)
        before = self.path.read_bytes()
        with patch("pickle.dump", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_bm25(self.index, self.metadata, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path, self.source})

    def test_offline_then_online_runner_without_building(self):
        output = io.StringIO()
        with patch.object(run_sparse_indexing, "__file__", str(self.root / "run_sparse_indexing.py")):
            with redirect_stdout(output):
                run_sparse_indexing.main()
        self.assertTrue(self.path.exists())
        output = io.StringIO()
        with patch.object(run_sparse_retrieval, "__file__", str(self.root / "run_sparse_retrieval.py")), \
             patch("sys.argv", ["run_sparse_retrieval.py", "hospital admissions"]), \
             patch("table_rag.sparse_retrieval.build_bm25", side_effect=AssertionError("Rebuilt!")), \
             patch("rank_bm25.BM25Okapi.__init__", side_effect=AssertionError("Rebuilt!")), \
             redirect_stdout(output):
            run_sparse_retrieval.main()
        self.assertIn("Loaded BM25 index with 3", output.getvalue())
        self.assertIn("1. b", output.getvalue())

    def test_search_runner_missing_index_has_actionable_error(self):
        error = io.StringIO()
        with patch.object(run_sparse_retrieval, "__file__", str(self.root / "run_sparse_retrieval.py")), \
             patch("sys.argv", ["run_sparse_retrieval.py", "hospital"]), redirect_stderr(error):
            with self.assertRaises(SystemExit) as caught:
                run_sparse_retrieval.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("run_sparse_indexing.py", error.getvalue())


if __name__ == "__main__":
    unittest.main()
