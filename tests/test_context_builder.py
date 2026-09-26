import json
from pathlib import Path
import tempfile
import unittest

from table_rag.context_builder import build_context, load_context_tables
from table_rag.sparse_retrieval import RetrievalResult


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        raw = self.root / "raw_tables/test"
        raw.mkdir(parents=True)
        (raw / "tbl_001.csv").write_text(
            "name,value,note\nA,10,\nB,25,highest\n", encoding="utf-8"
        )
        (raw / "tbl_002.csv").write_text(
            "name,count\nA,3\nB,4\n", encoding="utf-8"
        )
        self.manifest = self.root / "table_manifest.csv"
        self.manifest.write_text(
            "table_id,file_path,title,domain,row_count,column_count\n"
            "tbl_001,raw_tables/test/tbl_001.csv,Values,test,2,3\n"
            "tbl_002,raw_tables/test/tbl_002.csv,Counts,test,2,2\n",
            encoding="utf-8",
        )
        self.results = [
            RetrievalResult("tbl_001", 1, 0.9),
            RetrievalResult("tbl_002", 2, 0.7),
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_complete_ordered_context(self):
        package = build_context(
            "Which name has the highest value?",
            self.results,
            self.manifest,
            self.root,
            top_k=2,
        )
        self.assertEqual(package.table_ids, ("tbl_001", "tbl_002"))
        payload = json.loads(package.context_text)
        self.assertEqual(payload["question"], "Which name has the highest value?")
        self.assertEqual(payload["tables"][0]["columns"], ["name", "value", "note"])
        self.assertEqual(payload["tables"][0]["rows"], [
            ["A", "10", ""], ["B", "25", "highest"],
        ])
        self.assertEqual(payload["tables"][1]["retrieval_rank"], 2)

    def test_top_k_limits_tables_without_dropping_rows(self):
        tables = load_context_tables(
            self.results, self.manifest, self.root, top_k=1,
        )
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0].rows), 2)

    def test_rejects_unknown_table_id(self):
        with self.assertRaisesRegex(ValueError, "absent from the manifest"):
            build_context(
                "Question", [RetrievalResult("tbl_999", 1, 1.0)],
                self.manifest, self.root,
            )

    def test_rejects_silent_context_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceed the context"):
            build_context(
                "Question", self.results, self.manifest, self.root,
                max_characters=20,
            )

    def test_requires_consecutive_ranks(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            build_context(
                "Question", [RetrievalResult("tbl_001", 2, 0.9)],
                self.manifest, self.root,
            )


if __name__ == "__main__":
    unittest.main()
