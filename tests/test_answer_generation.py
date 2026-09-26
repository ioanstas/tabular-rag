from pathlib import Path
import tempfile
import unittest

from table_rag.answer_generation import (
    AnswerPayload,
    GeneratedAnswer,
    generate_answer,
    load_answer_instructions,
    validate_answer_payload,
)
from table_rag.context_builder import ContextPackage, ContextTable


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {"content": self.payload.model_dump_json()},
            "created_at": "2026-09-21T12:00:00Z",
        }


class AnswerGenerationTests(unittest.TestCase):
    def setUp(self):
        table = ContextTable(
            table_id="tbl_001",
            title="Values",
            domain="test",
            source_path="raw_tables/test/tbl_001.csv",
            retrieval_rank=1,
            retrieval_score=0.9,
            columns=("name", "value"),
            rows=(("A", "10"), ("B", "25")),
        )
        self.context = ContextPackage(
            question="Which name has the highest value?",
            tables=(table,),
            context_text='{"question":"Which name has the highest value?"}',
        )
        self.payload = AnswerPayload(
            status="answered",
            answer="B has the highest value, 25. [tbl_001]",
            cited_table_ids=["tbl_001"],
            calculation_or_evidence="Compared the value column across all rows.",
        )

    def test_calls_local_ollama_with_schema_and_validates_citation(self):
        client = FakeClient(self.payload)
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.md"
            prompt.write_text("Use only supplied evidence.", encoding="utf-8")
            answer = generate_answer(
                self.context,
                model="test-model",
                num_context=16_384,
                max_output_tokens=500,
                prompt_path=prompt,
                client=client,
            )
        self.assertIsInstance(answer, GeneratedAnswer)
        self.assertEqual(answer.cited_table_ids, ("tbl_001",))
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.response_id, "2026-09-21T12:00:00Z")
        call = client.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["format"], AnswerPayload.model_json_schema())
        self.assertFalse(call["think"])
        self.assertFalse(call["stream"])
        self.assertEqual(call["options"], {
            "temperature": 0,
            "num_ctx": 16_384,
            "num_predict": 500,
        })
        self.assertIn("Use only supplied evidence.", call["messages"][0]["content"])
        self.assertIn(
            "Return JSON matching this exact schema",
            call["messages"][0]["content"],
        )
        self.assertEqual(call["messages"][1]["content"], self.context.context_text)

    def test_renders_missing_inline_citations_from_declared_ids(self):
        payload = self.payload.model_copy(update={"answer": "B has value 25."})
        answer = generate_answer(self.context, client=FakeClient(payload))
        self.assertEqual(answer.text, "B has value 25. [tbl_001]")

    def test_generated_unknown_citations_are_still_rejected(self):
        payload = self.payload.model_copy(update={
            "answer": "B has value 25.", "cited_table_ids": ["tbl_999"],
        })
        with self.assertRaisesRegex(ValueError, "outside its context"):
            generate_answer(self.context, client=FakeClient(payload))

    def test_rejects_missing_citation_for_answered_status(self):
        payload = AnswerPayload(
            status="answered",
            answer="B has the highest value.",
            cited_table_ids=[],
            calculation_or_evidence="Compared all values.",
        )
        with self.assertRaisesRegex(ValueError, "must cite"):
            validate_answer_payload(payload, self.context.table_ids)

    def test_accepts_uncited_insufficient_evidence(self):
        payload = AnswerPayload(
            status="insufficient_evidence",
            answer="The supplied table does not contain dates.",
            cited_table_ids=[],
            calculation_or_evidence="No date column is present.",
        )
        self.assertEqual(
            validate_answer_payload(payload, self.context.table_ids), (),
        )

    def test_rejects_citation_outside_context(self):
        payload = AnswerPayload(
            status="answered",
            answer="The value is 25. [tbl_999]",
            cited_table_ids=["tbl_999"],
            calculation_or_evidence="Compared all values.",
        )
        with self.assertRaisesRegex(ValueError, "outside its context"):
            validate_answer_payload(payload, self.context.table_ids)

    def test_rejects_inline_and_declared_citation_mismatch(self):
        payload = AnswerPayload(
            status="answered",
            answer="The value is 25. [tbl_001]",
            cited_table_ids=[],
            calculation_or_evidence="Compared all values.",
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_answer_payload(payload, self.context.table_ids)

    def test_rejects_invalid_local_runtime_setting(self):
        with self.assertRaisesRegex(ValueError, "num_context"):
            generate_answer(self.context, num_context=0, client=FakeClient(self.payload))

    def test_rejects_empty_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.md"
            prompt.write_text("  ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt is empty"):
                load_answer_instructions(prompt)


if __name__ == "__main__":
    unittest.main()
