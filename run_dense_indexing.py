"""Build or reuse the offline local MiniLM dense index."""
import argparse
from pathlib import Path
import numpy as np

from table_rag.dense_index import (
    build_dense_index, load_dense_index, load_embedding_model, save_dense_index,
)
from table_rag.summaries import load_table_summaries, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Re-encode even if the saved index is current")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached model files only")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    root = Path(__file__).resolve().parent
    source = root / "artifacts/table_summaries.jsonl"
    output = root / "artifacts/dense_index.npz"
    try:
        source_hash = sha256_file(source)
        summaries = load_table_summaries(
            source, root / "dataset", root / "prompts/table_summary_v1.md",
        )
        if output.exists() and not args.rebuild:
            try:
                index = load_dense_index(output, summary_file_sha256=source_hash)
                if index.table_ids != tuple(summary.table_id for summary in summaries):
                    raise ValueError("Saved dense table IDs differ from the summary dataset")
            except ValueError as exc:
                print(f"Rebuilding existing dense index: {exc}")
            else:
                print(f"Reused dense index: {index.vectors.shape[0]} tables x {index.vectors.shape[1]} dimensions.")
                return
        print(f"Validated {len(summaries)} summaries. Loading the local embedding model...", flush=True)
        model = load_embedding_model(local_files_only=args.local_files_only)
        index = build_dense_index(summaries, source_hash, model, batch_size=args.batch_size)
        if sha256_file(source) != source_hash:
            raise ValueError("Summaries changed during embedding; run indexing again")
        save_dense_index(index, output)
        restored = load_dense_index(output, summary_file_sha256=source_hash)
        if restored.table_ids != index.table_ids or not np.array_equal(restored.vectors, index.vectors):
            raise ValueError("Dense index round-trip verification failed")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved dense index: {index.vectors.shape[0]} tables x {index.vectors.shape[1]} dimensions.")
    print(f"Longest summary: {index.metadata['max_observed_tokens']} tokens; no truncation.")
    print(f"Verified exact save/load equality: {output}")


if __name__ == "__main__":
    main()
