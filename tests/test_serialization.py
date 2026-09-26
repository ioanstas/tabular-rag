"""Focused checks for lossless, deterministic table documents."""
import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from table_rag.ingestion import LoadedTable
from table_rag.serialization import save_documents, serialize_table, serialize_tables


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.table = LoadedTable(
            table_id="tbl_002",
            data=pd.DataFrame(
                [["001", "1.00", "Αθήνα", ""],
                 ["002", "0.010", 'A "quote"\nrow: [x]\\end', " NA "]],
                columns=["code", "value", "note", "empty"],
                index=[9, 2], dtype="string",
            ),
            metadata={"title": 'Monthly "counts"\nΕλλάδα', "domain": "healthcare"},
        )
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nested" / "documents.jsonl"

    def test_exact_format_and_text_preservation(self):
        expected = (
            'table_id: "tbl_002"\n'
            'title: "Monthly \\"counts\\"\\nΕλλάδα"\n'
            'domain: "healthcare"\n'
            'columns: ["code", "value", "note", "empty"]\n'
            'row: ["001", "1.00", "Αθήνα", ""]\n'
            'row: ["002", "0.010", "A \\"quote\\"\\nrow: [x]\\\\end", " NA "]'
        )
        self.assertEqual(serialize_table(self.table), expected)

    def test_column_order_and_escaped_names(self):
        self.table.data = self.table.data[["empty", "note", "value", "code"]]
        self.table.data.columns = ["empty", 'note"\n', "value", "code"]
        lines = serialize_table(self.table).split("\n")
        self.assertEqual(json.loads(lines[3].split(": ", 1)[1]),
                         ["empty", 'note"\n', "value", "code"])
        rows = [json.loads(line.split(": ", 1)[1]) for line in lines[4:]]
        self.assertEqual(rows, self.table.data.values.tolist())

    def test_zero_rows_preserves_columns(self):
        self.table.data = self.table.data.iloc[:0]
        text = serialize_table(self.table)
        self.assertEqual(len(text.split("\n")), 4)
        self.assertTrue(text.endswith('columns: ["code", "value", "note", "empty"]'))

    def test_batch_order_and_exact_keys(self):
        other = LoadedTable("tbl_001", self.table.data, self.table.metadata)
        documents = serialize_tables({"tbl_002": self.table, "tbl_001": other})
        self.assertEqual([d["table_id"] for d in documents], ["tbl_002", "tbl_001"])
        for document in documents:
            self.assertEqual(list(document), ["table_id", "serialized_text"])
        self.assertEqual(serialize_tables({}), [])

    def test_mismatched_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            serialize_tables({"wrong": self.table})

    def test_jsonl_round_trip_and_identical_bytes(self):
        documents = serialize_tables({"tbl_002": self.table})
        # Input key order must not change artifact key order.
        reversed_keys = [dict(reversed(list(documents[0].items())))]
        save_documents(reversed_keys, str(self.path))
        first = self.path.read_bytes()
        expected = (json.dumps(documents[0], ensure_ascii=False) + "\n").encode("utf-8")
        self.assertEqual(first, expected)
        self.assertEqual([json.loads(line) for line in first.splitlines()], documents)
        save_documents(documents, self.path)
        self.assertEqual(self.path.read_bytes(), first)
        save_documents([], self.path)
        self.assertEqual(self.path.read_bytes(), b"")

    def test_duplicate_rejected_before_overwrite(self):
        documents = serialize_tables({"tbl_002": self.table})
        save_documents(documents, self.path)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            save_documents(documents * 2, self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_fields_rejected_before_writing(self):
        good = {"table_id": "tbl_001", "serialized_text": "text"}
        for key in good:
            for value in ("", " \n", None, 1):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "non-empty string"):
                        save_documents([good, {**good, key: value}], self.path)
                    self.assertFalse(self.path.parent.exists())
            invalid = good.copy()
            del invalid[key]
            with self.assertRaises(ValueError):
                save_documents([invalid], self.path)

    def test_imports_perform_no_work(self):
        import table_rag.serialization as serialization
        import run_serialization
        with patch("table_rag.ingestion.load_manifest") as loader, \
                patch.object(Path, "write_text") as writer, redirect_stdout(io.StringIO()) as output:
            importlib.reload(serialization)
            importlib.reload(run_serialization)
        loader.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
