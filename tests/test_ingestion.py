"""Small CSV fixtures exercise ingestion without external services."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from table_rag.ingestion import load_all_tables, load_manifest, load_table


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "tables" / "one.csv"
        self.source.parent.mkdir()
        self.source.write_text(
            "code,date,value,note\n001,2025-01,1.00,\n002,2025-02,,NA\n",
            encoding="utf-8",
        )
        self.entry = pd.Series({
            "table_id": "tbl_001", "file_path": "tables/one.csv",
            "title": "Example", "domain": "testing",
            "row_count": 2, "column_count": 4, "source_name": "Fixture",
        })

    def test_text_metadata_and_order_preserved(self):
        table = load_table(self.entry, self.root)
        self.assertEqual(table.table_id, "tbl_001")
        self.assertEqual(table.metadata, self.entry.to_dict())
        self.assertEqual(table.data.columns.tolist(), ["code", "date", "value", "note"])
        self.assertEqual(table.data.values.tolist(), [
            ["001", "2025-01", "1.00", ""], ["002", "2025-02", "", "NA"]
        ])
        table.metadata["title"] = "Changed"
        self.assertEqual(self.entry["title"], "Example")

    def test_missing_source_has_context(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError) as error:
            load_table(self.entry, self.root)
        self.assertIn("tbl_001", str(error.exception))
        self.assertIn(str(self.source.resolve()), str(error.exception))

    def test_unreadable_csv_has_context(self):
        for contents in ('a,b\n"unterminated,b\n', ''):
            with self.subTest(contents=contents):
                self.source.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError) as error:
                    load_table(self.entry, self.root)
                self.assertIn("tbl_001", str(error.exception))
                self.assertIn(str(self.source.resolve()), str(error.exception))

    def test_dimensions_must_match(self):
        for column in ("row_count", "column_count"):
            with self.subTest(column=column):
                entry = self.entry.copy()
                entry[column] = 99
                with self.assertRaisesRegex(ValueError, "expected .*actual"):
                    load_table(entry, self.root)

    def test_batch_preserves_order(self):
        second = self.entry.copy()
        second["table_id"] = "tbl_002"
        manifest = pd.DataFrame([second, self.entry])
        tables = load_all_tables(manifest, self.root)
        self.assertEqual(list(tables), ["tbl_002", "tbl_001"])
        self.assertEqual(tables["tbl_001"].data.shape, (2, 4))

    def test_duplicate_ids_rejected_before_loading(self):
        manifest = pd.DataFrame([self.entry, self.entry])
        with patch("table_rag.ingestion.load_table") as loader:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_all_tables(manifest, self.root)
            loader.assert_not_called()

    def test_batch_stops_on_failure(self):
        second = self.entry.copy()
        second["table_id"] = "tbl_002"
        manifest = pd.DataFrame([self.entry, second])
        with patch("table_rag.ingestion.load_table", side_effect=ValueError("bad CSV")) as loader:
            with self.assertRaisesRegex(ValueError, "bad CSV"):
                load_all_tables(manifest, self.root)
            self.assertEqual(loader.call_count, 1)

    def test_manifest_to_batch(self):
        path = self.root / "table_manifest.csv"
        pd.DataFrame([self.entry]).to_csv(path, index=False)
        manifest = load_manifest(path)
        self.assertEqual(manifest.iloc[0]["row_count"], 2)
        self.assertEqual(list(load_all_tables(manifest, self.root)), ["tbl_001"])

    def test_invalid_manifests(self):
        good = pd.DataFrame([self.entry])
        blank = good.copy()
        blank.loc[0, "title"] = " "
        bad_count = good.copy()
        bad_count["row_count"] = "-1"
        cases = [good.iloc[:0], good.drop(columns=["domain"]),
                 pd.concat([good, good]), blank, bad_count]
        path = self.root / "table_manifest.csv"
        for frame in cases:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_manifest(path)


if __name__ == "__main__":
    unittest.main()
