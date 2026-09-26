"""Verify dense artifact identity, freshness, and encoding boundaries."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from table_rag.dense_index import (
    DIMENSION, MAX_SEQUENCE_LENGTH, DenseIndex, build_dense_index,
    embed_texts, load_dense_index, save_dense_index,
)
from table_rag.summaries import TableSummary, sha256_file
import run_dense_indexing


def fake_encoder():
    model = Mock()
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.tokenizer.side_effect = lambda texts, **kwargs: {
        "input_ids": [list(range(len(text.split()) + 2)) for text in texts]
    }
    model.encode.side_effect = lambda texts, **kwargs: np.eye(
        len(texts), DIMENSION, dtype=np.float32,
    )
    return model


class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "artifacts").mkdir()
        self.source = self.root / "artifacts/table_summaries.jsonl"
        self.source.write_text("fixture input", encoding="utf-8")
        self.source_hash = sha256_file(self.source)
        self.path = self.root / "artifacts/dense_index.npz"
        self.summaries = [
            TableSummary("tbl_b", "Hospital admissions", "a" * 64, "b" * 64,
                         "assistant_checked_pending_human_review"),
            TableSummary("tbl_a", "Hotel bookings", "c" * 64, "d" * 64,
                         "assistant_checked_pending_human_review"),
        ]
        self.model = fake_encoder()
        self.index = build_dense_index(self.summaries, self.source_hash, self.model)

    def test_original_text_and_id_order_reach_encoder(self):
        self.assertEqual(self.index.table_ids, ("tbl_b", "tbl_a"))
        self.assertEqual(self.model.encode.call_args.args[0], ["Hospital admissions", "Hotel bookings"])
        self.assertTrue(self.model.encode.call_args.kwargs["normalize_embeddings"])
        np.testing.assert_array_equal(self.index.vectors, np.eye(2, DIMENSION, dtype=np.float32))

    def test_round_trip_needs_no_model(self):
        save_dense_index(self.index, self.path)
        with patch("table_rag.dense_index.load_embedding_model", side_effect=AssertionError("Loaded model")):
            loaded = load_dense_index(self.path, summary_file_sha256=self.source_hash)
        self.assertEqual(loaded.table_ids, self.index.table_ids)
        self.assertEqual(loaded.metadata, self.index.metadata)
        np.testing.assert_array_equal(loaded.vectors, self.index.vectors)

    def test_oversized_text_is_rejected_before_encoding(self):
        model = fake_encoder()
        with self.assertRaisesRegex(ValueError, "exceed"):
            embed_texts(model, ["word " * MAX_SEQUENCE_LENGTH])
        model.encode.assert_not_called()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            embed_texts(model, ["   "])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            embed_texts(model, ["text"], batch_size=0)

    def test_bad_encoder_outputs_are_rejected(self):
        for output in (
            np.zeros((2, DIMENSION), dtype=np.float32),
            np.full((2, DIMENSION), np.nan, dtype=np.float32),
            np.ones((2, 10), dtype=np.float32),
        ):
            model = fake_encoder()
            model.encode.side_effect = None
            model.encode.return_value = output
            with self.assertRaises(ValueError):
                build_dense_index(self.summaries, self.source_hash, model)

    def test_empty_duplicate_inputs_are_rejected(self):
        for records in ([], self.summaries[:1] * 2):
            with self.assertRaises(ValueError):
                build_dense_index(records, self.source_hash, fake_encoder())

    def test_stale_summary_and_wrong_model_metadata(self):
        save_dense_index(self.index, self.path)
        with self.assertRaisesRegex(ValueError, "stale"):
            load_dense_index(self.path, summary_file_sha256="0" * 64)
        for field, value in (
            ("model_id", "other"), ("model_revision", "other"),
            ("dimension", 42), ("encoding", {}),
            ("embedding_library_versions", {}),
        ):
            metadata = {**self.index.metadata, field: value}
            np.savez_compressed(self.path, vectors=self.index.vectors,
                                table_ids=np.array(self.index.table_ids),
                                metadata=np.array(json.dumps(metadata)))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Incompatible"):
                load_dense_index(self.path)

    def test_reordered_ids_or_vectors_fail_checksum(self):
        for ids, vectors in (
            (self.index.table_ids[::-1], self.index.vectors),
            (self.index.table_ids, self.index.vectors[::-1]),
        ):
            np.savez_compressed(self.path, vectors=vectors, table_ids=np.array(ids),
                                metadata=np.array(json.dumps(self.index.metadata)))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dense_index(self.path)

    def test_missing_corrupt_or_pickle_arrays_fail_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "run_dense_indexing"):
            load_dense_index(self.path)
        self.path.write_bytes(b"not an npz")
        with self.assertRaisesRegex(ValueError, "run_dense_indexing"):
            load_dense_index(self.path)
        np.savez_compressed(self.path, vectors=self.index.vectors,
                            table_ids=np.array(self.index.table_ids, dtype=object),
                            metadata=np.array(json.dumps(self.index.metadata)))
        with self.assertRaisesRegex(ValueError, "run_dense_indexing"):
            load_dense_index(self.path)

    def test_failed_save_preserves_existing_artifact(self):
        save_dense_index(self.index, self.path)
        before = self.path.read_bytes()
        with patch("numpy.savez_compressed", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_dense_index(self.index, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path, self.source})

    def test_runner_reuses_current_index_and_rebuilds_when_requested(self):
        save_dense_index(self.index, self.path)
        for arguments, expected_calls in (([], 0), (["--rebuild"], 1)):
            output = io.StringIO()
            with patch.object(run_dense_indexing, "__file__", str(self.root / "run_dense_indexing.py")), \
                 patch("sys.argv", ["run_dense_indexing.py"] + arguments), \
                 patch.object(run_dense_indexing, "load_table_summaries", return_value=self.summaries), \
                 patch.object(run_dense_indexing, "load_embedding_model", return_value=fake_encoder()) as loader, \
                 redirect_stdout(output):
                run_dense_indexing.main()
            self.assertEqual(loader.call_count, expected_calls)
            self.assertIn("Reused" if not arguments else "Saved", output.getvalue())


if __name__ == "__main__":
    unittest.main()
