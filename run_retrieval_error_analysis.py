"""Create retrieval error tables and PowerPoint-ready charts from saved results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from table_rag.error_analysis import (
    dense_weighted_comparison, failure_summary, grouped_metrics, load_query_results,
)


METHODS = ("BM25", "Dense", "RRF", "Weighted a=0.8")
COLORS = {
    "BM25": "#D97706",
    "Dense": "#2563EB",
    "RRF": "#7C3AED",
    "Weighted a=0.8": "#059669",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _add_title(axis, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", weight="bold", fontsize=18, pad=34)
    axis.text(
        0, 1.01, subtitle, transform=axis.transAxes,
        color="#4B5563", fontsize=10, va="bottom",
    )


def _style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _failure_chart(summary: list[dict], path: Path) -> None:
    test = {(row["method"]): row for row in summary if row["split"] == "test"}
    x = np.arange(len(METHODS))
    width = 0.23
    figure, axis = plt.subplots(figsize=(12, 6.75))
    for offset, (field, label, shade) in enumerate((
        ("top_1_failures", "Outside top 1", 0.0),
        ("top_3_failures", "Outside top 3", 0.18),
        ("top_5_failures", "Outside top 5", 0.36),
    )):
        values = [test[method][field] for method in METHODS]
        bars = axis.bar(
            x + (offset - 1) * width, values, width, label=label,
            color=[COLORS[method] for method in METHODS], alpha=1 - shade,
        )
        axis.bar_label(bars, padding=3, fontsize=10)
    axis.set_xticks(x, METHODS)
    axis.set_ylabel("Number of failed questions")
    _add_title(
        axis, "Retrieval failures on the held-out test set",
        "58 answerable questions; q002 and q003 excluded",
    )
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.set_ylim(0, max(test[m]["top_1_failures"] for m in METHODS) + 4)
    _style_axis(axis)
    figure.tight_layout()
    _save_figure(figure, path)


def _query_type_chart(rows: list[dict], path: Path) -> None:
    test = [row for row in rows if row["split"] == "test"]
    query_types = sorted({row["query_type"] for row in test})
    lookup = {(row["method"], row["query_type"]): row for row in test}
    x = np.arange(len(query_types))
    width = 0.19
    figure, axis = plt.subplots(figsize=(12, 6.75))
    for index, method in enumerate(METHODS):
        values = [lookup[(method, value)]["hit_rate_at_1"] for value in query_types]
        bars = axis.bar(
            x + (index - 1.5) * width, values, width,
            label=method, color=COLORS[method],
        )
        axis.bar_label(bars, labels=[f"{value:.0%}" for value in values],
                       padding=3, fontsize=9)
    axis.set_xticks(x, [value.title() for value in query_types])
    axis.set_ylabel("Hit@1")
    axis.set_ylim(0, 1.12)
    _add_title(
        axis, "Top-1 retrieval by question type",
        "Held-out test set; percentages show correct table ranked first",
    )
    axis.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _style_axis(axis)
    figure.tight_layout()
    _save_figure(figure, path)


def _domain_chart(rows: list[dict], path: Path) -> None:
    test = [row for row in rows if row["split"] == "test"]
    domains = sorted({row["domain"] for row in test})
    lookup = {(row["method"], row["domain"]): row for row in test}
    matrix = np.array([
        [lookup[(method, domain)]["hit_rate_at_1"] for domain in domains]
        for method in METHODS
    ])
    counts = {domain: lookup[(METHODS[0], domain)]["query_count"] for domain in domains}
    figure, axis = plt.subplots(figsize=(13, 6.75))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row_index in range(len(METHODS)):
        for column_index in range(len(domains)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index, row_index, f"{value:.0%}",
                ha="center", va="center",
                color="white" if value >= 0.72 else "#111827", weight="bold",
            )
    axis.set_xticks(
        range(len(domains)),
        [f"{domain.replace('_', ' ').title()}\n(n={counts[domain]})" for domain in domains],
        rotation=25, ha="right",
    )
    axis.set_yticks(range(len(METHODS)), METHODS)
    _add_title(
        axis, "Hit@1 by domain",
        "Held-out test set; subgroup sizes are shown under each domain",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    figure.tight_layout()
    _save_figure(figure, path)


def _case_studies(rows: list[dict], comparison: list[dict]) -> list[dict]:
    by_query = {}
    for row in rows:
        by_query.setdefault((row["split"], row["query_id"]), {})[row["method"]] = row
    cases = []
    semantic = [
        methods for (split, _), methods in by_query.items()
        if split == "test" and methods["Dense"]["query_type"] == "semantic"
        and methods["BM25"]["hit_at_1"] == 0 and methods["Dense"]["hit_at_1"] == 1
    ]
    if semantic:
        cases.append({"case": "BM25 fails; dense succeeds", **semantic[0]["Dense"],
                      "method_results": {m: semantic[0][m]["top_5_table_ids"] for m in METHODS}})
    improved = [row for row in comparison if row["split"] == "dev"
                and row["outcome"] == "weighted_improved"]
    if improved:
        methods = by_query[("dev", improved[0]["query_id"])]
        cases.append({"case": "Weighted fusion improves dense rank", **improved[0],
                      "method_results": {m: methods[m]["top_5_table_ids"] for m in METHODS}})
    difficult = [
        methods for (split, _), methods in by_query.items()
        if split == "test" and methods["Weighted a=0.8"]["hit_at_1"] == 0
    ]
    if difficult:
        cases.append({"case": "Difficult held-out test query", **difficult[0]["Weighted a=0.8"],
                      "method_results": {m: difficult[0][m]["top_5_table_ids"] for m in METHODS}})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path,
                        default=root / "artifacts/retrieval_query_results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts")
    args = parser.parse_args()
    try:
        rows = load_query_results(args.input)
        methods = {row["method"] for row in rows}
        if methods != set(METHODS):
            raise ValueError(f"Expected methods {METHODS}, found {sorted(methods)}")
        summary = failure_summary(rows)
        by_type = grouped_metrics(rows, "query_type")
        by_domain = grouped_metrics(rows, "domain")
        comparison = dense_weighted_comparison(rows, "Weighted a=0.8")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "retrieval_failure_summary.csv", summary)
        _write_csv(args.output_dir / "retrieval_by_query_type.csv", by_type)
        _write_csv(args.output_dir / "retrieval_by_domain.csv", by_domain)
        _write_csv(args.output_dir / "dense_weighted_comparison.csv", comparison)
        _failure_chart(summary, args.output_dir / "test_failures_by_cutoff.png")
        _query_type_chart(by_type, args.output_dir / "test_hit1_by_query_type.png")
        _domain_chart(by_domain, args.output_dir / "test_hit1_by_domain.png")
        cases = _case_studies(rows, comparison)
        comparison_counts = {}
        for split in ("dev", "test"):
            comparison_counts[split] = {
                outcome: sum(row["split"] == split and row["outcome"] == outcome
                             for row in comparison)
                for outcome in ("weighted_improved", "dense_improved", "same_relevant_rank")
            }
        report = {
            "input": str(args.input),
            "methods": list(METHODS),
            "dense_weighted_relevant_rank_comparison": comparison_counts,
            "case_studies": cases,
            "presentation_files": [
                "test_failures_by_cutoff.png",
                "test_hit1_by_query_type.png",
                "test_hit1_by_domain.png",
            ],
        }
        path = args.output_dir / "retrieval_error_analysis.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Analysed {len(rows)} query-method rows.")
    for row in summary:
        if row["split"] == "test":
            print(f"{row['method']}: top-1 failures={row['top_1_failures']}, "
                  f"top-3={row['top_3_failures']}, top-5={row['top_5_failures']}")
    print(f"Saved tables, charts, and case studies in {args.output_dir}")


if __name__ == "__main__":
    main()
