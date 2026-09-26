# Reproduce the results

Extract the ZIP and open a terminal in `tabular_rag_code`.
Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
Tested on Ubuntu/WSL with Python 3.13.

## 1. Install

```bash
uv sync --locked
```

## 2. Build the indexes

The CSV tables and prepared summaries are included.

```bash
uv run python run_ingestion.py
uv run python run_serialization.py
uv run python run_sparse_indexing.py
uv run python run_summary_validation.py
uv run python run_dense_indexing.py --rebuild
```

The first dense run downloads MiniLM.

## 3. Reproduce retrieval results

```bash
uv run python run_retrieval_evaluation.py
uv run python run_additional_retrieval_evaluation.py
uv run python run_retrieval_error_analysis.py
```

Results and charts are saved in `artifacts/`. The first command selects alpha
on the development set; the additional evaluation uses the fixed alpha of 0.8.

## 4. Reproduce answers with and without RAG

Install and start [Ollama](https://docs.ollama.com/quickstart). In its terminal:

```bash
ollama pull qwen3.5:9b
```

If Python runs in WSL and Ollama runs on Windows, run
`.\start_ollama_wsl.ps1` in PowerShell and copy its printed `export` command
into the WSL terminal.

Back in the project terminal, preserve the included results and run:

```bash
uv run python -c "from pathlib import Path; Path('artifacts/answer_comparison').rename('artifacts/answer_comparison_submitted')"
uv run python run_answer_comparison.py
```

Rename only once. New answers and scores are saved in
`artifacts/answer_comparison/`. This makes 24 local model calls and may take a while.
Generated answers can vary between runs.
