"""Validate the prepared summary dataset before embedding it."""
from pathlib import Path
from table_rag.summaries import load_table_summaries


def main() -> None:
    root = Path(__file__).resolve().parent
    summaries = load_table_summaries(
        root / "artifacts" / "table_summaries.jsonl",
        root / "dataset",
        root / "prompts" / "table_summary_v1.md",
    )
    pending = sum(item.review_status != "human_approved" for item in summaries)
    print(f"Validated {len(summaries)} unique, current table summaries.")
    print(f"Human review pending: {pending}. Structural checks do not certify prose correctness.")
    print("\nFirst summary:")
    print(summaries[0].summary_text)


if __name__ == "__main__":
    main()
