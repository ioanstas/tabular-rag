"""Check that summary artifacts remain aligned with their source tables."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from table_rag.ingestion import load_manifest, load_all_tables
from table_rag.summaries import (
    GENERATION_METHOD, PROMPT_VERSION, SCHEMA_VERSION, load_table_summaries,
    metadata_sha256, sha256_file, sha256_text, source_profile,
)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompt = self.root / "prompt.md"
        self.prompt.write_text("Describe table contents.", encoding="utf-8")
        self.path = self.root / "summaries.jsonl"
        self.manifest = self.root / "table_manifest.csv"
        fields = ["table_id", "file_path", "title", "domain", "row_count",
                  "column_count", "date_start", "date_end", "row_granularity"]
        entries = []
        for table_id in ("b", "a"):
            (self.root / f"{table_id}.csv").write_text(
                "name,report_month,candidates_submitted\nX,2025-01,20\nY,2025-12,100\n",
                encoding="utf-8",
            )
            entries.append(dict(zip(fields, [
                table_id, f"{table_id}.csv", "Example table", "example", "2",
                "3", "2025-01", "2025-06", "name-month",
            ])))
        with self.manifest.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
        tables = load_all_tables(load_manifest(self.manifest), self.root)
        self.records = []
        for table_id, table in tables.items():
            text = "Synthetic example observations."
            self.records.append({
                "schema_version": SCHEMA_VERSION, "table_id": table_id,
                "summary_text": text, "summary_sha256": sha256_text(text),
                "source_sha256": sha256_file(self.root / table.metadata["file_path"]),
                "metadata_sha256": metadata_sha256(table.metadata),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": sha256_file(self.prompt),
                "generation_method": GENERATION_METHOD,
                "review_status": "assistant_checked_pending_human_review",
                "source_profile": source_profile(table),
            })
        self.write_records()

    def write_records(self):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")

    def load(self):
        return load_table_summaries(self.path, self.root, self.prompt)

    def test_round_trip_returns_manifest_order(self):
        self.records.reverse()
        self.write_records()
        self.assertEqual([r.table_id for r in self.load()], ["b", "a"])

    def test_profile_uses_observed_dates_and_avoids_candidate_name_false_match(self):
        profile = self.records[0]["source_profile"]
        self.assertEqual(set(profile["observed_time_ranges"]), {"report_month"})
        self.assertEqual(profile["observed_time_ranges"]["report_month"]["max"], "2025-12")
        self.assertEqual(profile["warnings"][0]["observed"], ["2025-01", "2025-12"])

    def test_missing_duplicate_and_unknown_ids(self):
        original = list(self.records)
        for records, error in [
            (original[:1], "Missing"),
            (original + original[:1], "Duplicate"),
            ([{**original[0], "table_id": "unknown"}, original[1]], "Unknown"),
        ]:
            with self.subTest(error=error):
                self.records = records
                self.write_records()
                with self.assertRaisesRegex(ValueError, error):
                    self.load()

    def test_changed_source_and_metadata_are_rejected(self):
        source = self.root / "b.csv"
        content = source.read_text()
        source.write_text(content.replace(",20", ",21"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale summary source"):
            self.load()
        source.write_text(content, encoding="utf-8")
        self.manifest.write_text(self.manifest.read_text().replace("Example table", "New title"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale summary metadata"):
            self.load()

    def test_changed_prompt_is_rejected(self):
        self.prompt.write_text("Changed instructions.", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "instructions changed"):
            self.load()

    def test_invalid_content_and_provenance(self):
        original = dict(self.records[0])
        for key, value in [
            ("summary_text", ""), ("summary_text", "Altered text"),
            ("schema_version", 99), ("review_status", "unverified"),
            ("generation_method", "unknown"), ("source_profile", {}),
            ("prompt_version", "v999"),
        ]:
            with self.subTest(field=key):
                self.records[0] = {**original, key: value}
                self.write_records()
                with self.assertRaises(ValueError):
                    self.load()

    def test_malformed_json_and_nonobject_records(self):
        for content in ("bad json\n", "[]\n"):
            self.path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.load()


if __name__ == "__main__":
    unittest.main()
