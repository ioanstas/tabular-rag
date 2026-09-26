"""Check ranking, query-only encoding, and saved-index search boundaries."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from table_rag.dense_index import DIMENSION, MAX_SEQUENCE_LENGTH, build_dense_index, save_dense_index
from table_rag.dense_retrieval import search_dense
from table_rag.summaries import TableSummary, sha256_file
import run_dense_retrieval


def encoder(vectors):
    model = Mock()
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [list(range(len(text.split()) + 2)) for text in texts]
    }
    model.encode.return_value = vectors
    return model


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        # Deliberately unsorted IDs; two equal scores and one negative score.
        vectors = np.zeros((3, DIMENSION), dtype=np.float32)
        vectors[0, 0] = vectors[1, 0] = 1
        vectors[2, 0] = -1
        summaries = [
            TableSummary(table_id, "summary text", "a" * 64, "b" * 64, "human_approved")
            for table_id in ("tbl_b", "tbl_a", "tbl_c")
        ]
        self.index = build_dense_index(summaries, "a" * 64, encoder(vectors))
        self.model = encoder(vectors[:1].copy())

    def test_ranking_ties_negative_scores_and_query_only_encoding(self):
        before = self.index.vectors.copy()
        results = search_dense(self.index, self.model, "hospital admissions?", 10)
        self.assertEqual([r.table_id for r in results], ["tbl_a", "tbl_b", "tbl_c"])
        self.assertEqual([r.rank for r in results], [1, 2, 3])
        self.assertEqual([r.score for r in results], [1.0, 1.0, -1.0])
        self.model.encode.assert_called_once()
        self.assertEqual(self.model.encode.call_args.args[0], ["hospital admissions?"])
        self.assertTrue(self.model.encode.call_args.kwargs["normalize_embeddings"])
        np.testing.assert_array_equal(before, self.index.vectors)

    def test_changed_question_vector_changes_winner_and_top_k_limits(self):
        self.model.encode.return_value = -self.model.encode.return_value
        results = search_dense(self.index, self.model, "different question", 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].table_id, "tbl_c")

    def test_bad_inputs_fail_before_inference(self):
        for question in ("", "  ", None, 42):
            with self.subTest(question=question), self.assertRaises(ValueError):
                search_dense(self.index, self.model, question)
        for top_k in (0, -1, True, 1.5, "5"):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                search_dense(self.index, self.model, "question", top_k)
        with self.assertRaisesRegex(ValueError, "exceed"):
            search_dense(self.index, self.model, "word " * MAX_SEQUENCE_LENGTH)
        self.model.encode.assert_not_called()

    def test_corrupted_mapping_fails_before_inference(self):
        self.index.metadata["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            search_dense(self.index, self.model, "question")
        self.model.encode.assert_not_called()

    def test_runner_loads_saved_vectors_and_rejects_stale_summaries(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = artifacts / "table_summaries.jsonl"
            source.write_text("summary fixture", encoding="utf-8")
            self.index.metadata["summary_file_sha256"] = sha256_file(source)
            path = artifacts / "dense_index.npz"
            save_dense_index(self.index, path)
            before = path.read_bytes()
            with patch.object(run_dense_retrieval, "__file__", str(root / "run_dense_retrieval.py")), \
                 patch("sys.argv", ["run_dense_retrieval.py", "question", "--top-k", "1", "--local-files-only"]), \
                 patch.object(run_dense_retrieval, "load_embedding_model", return_value=self.model) as loader:
                output = io.StringIO()
                with redirect_stdout(output):
                    run_dense_retrieval.main()
                loader.assert_called_once_with(local_files_only=True)
                self.assertIn("1. tbl_a", output.getvalue())
                self.assertNotIn("2.", output.getvalue())
                self.assertEqual(before, path.read_bytes())
                source.write_text("changed", encoding="utf-8")
                loader.reset_mock()
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                    run_dense_retrieval.main()
                self.assertEqual(caught.exception.code, 1)
                self.assertIn("stale", error.getvalue())
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
