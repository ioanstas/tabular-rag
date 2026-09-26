"""Retrieve tables, build exact CSV context, and generate a cited answer."""
import argparse
import math
from pathlib import Path

from table_rag.answer_generation import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_NUM_CONTEXT,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    generate_answer,
)
from table_rag.context_builder import build_context
from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.hybrid_retrieval import fuse_weighted, retrieve_candidates
from table_rag.sparse_retrieval import load_bm25
from table_rag.summaries import sha256_file


DEFAULT_ALPHA = 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Quote the complete question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-context-characters", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--num-context", type=int, default=DEFAULT_NUM_CONTEXT)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--save-context", type=Path,
        help="Optional path for saving the exact context sent to the answer model",
    )
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("question must be non-empty text")
    if (
        args.top_k < 1
        or args.max_context_characters < 1
        or args.max_output_tokens < 1
        or args.num_context < 1
        or args.request_timeout_seconds < 1
    ):
        parser.error(
            "table, context, model context, output, and timeout limits must be positive"
        )
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    dataset = root / "dataset"
    try:
        sparse_index = load_bm25(
            artifacts / "bm25_index.pkl",
            source_path=artifacts / "sparse_documents.jsonl",
        )
        dense_index = load_dense_index(
            artifacts / "dense_index.npz",
            summary_file_sha256=sha256_file(artifacts / "table_summaries.jsonl"),
        )
        embedding_model = load_embedding_model(
            local_files_only=args.local_files_only,
        )
        sparse, dense = retrieve_candidates(
            sparse_index, dense_index, embedding_model, args.question,
        )
        results = fuse_weighted(
            sparse, dense, top_k=args.top_k, alpha=args.alpha,
        )
        context = build_context(
            args.question,
            results,
            dataset / "table_manifest.csv",
            dataset,
            top_k=args.top_k,
            max_characters=args.max_context_characters,
        )
        if args.save_context:
            args.save_context.parent.mkdir(parents=True, exist_ok=True)
            args.save_context.write_text(context.context_text + "\n", encoding="utf-8")
        answer = generate_answer(
            context,
            model=args.answer_model,
            base_url=args.ollama_url,
            num_context=args.num_context,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.request_timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print("Retrieved evidence tables:")
    for table in context.tables:
        print(
            f"{table.retrieval_rank}. {table.table_id}  "
            f"score={table.retrieval_score:.6f}  {table.title}"
        )
    print(f"\nAnswer model: {answer.model}")
    print(f"Status: {answer.status}")
    print(answer.text)
    print(f"Evidence: {answer.calculation_or_evidence}")
    citations = ", ".join(answer.cited_table_ids) or "none"
    print(f"Cited tables: {citations}")


if __name__ == "__main__":
    main()
