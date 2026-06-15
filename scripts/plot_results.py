#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to render the paper figures. "
            "Install it with 'python3 -m pip install matplotlib'."
        ) from exc
    return plt


def load_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def render_bar_chart(plt, labels, values, ylabel, title, output_path: Path, color):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render paper-ready plots from compare_results.csv.")
    parser.add_argument("--input", required=True, help="Path to compare_results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit("No benchmark rows found in input CSV.")

    plt = require_matplotlib()

    choreo_success_rate = 100.0 * sum(r["choreo_status"] == "success" for r in rows) / len(rows)
    mlir_success_rate = 100.0 * sum(r["mlir_status"] == "success" for r in rows) / len(rows)
    render_bar_chart(
        plt,
        ["Choreo", "MLIR"],
        [choreo_success_rate, mlir_success_rate],
        "Successful cases (%)",
        "Symbolic-shape benchmark slice success rate",
        output_dir / "success_rate",
        ["#2f6bff", "#ff7f0e"],
    )

    choreo_symbolic_rate = 100.0 * sum(r["choreo_class"] == "symbolic" for r in rows) / len(rows)
    mlir_dynamic_rate = 100.0 * sum(r["mlir_class"] == "dynamic" for r in rows) / len(rows)
    render_bar_chart(
        plt,
        ["Choreo symbolic", "MLIR dynamic"],
        [choreo_symbolic_rate, mlir_dynamic_rate],
        "Cases (%)",
        "Dynamic/symbolic cases retained after analysis",
        output_dir / "dynamic_symbolic_rate",
        ["#4daf4a", "#984ea3"],
    )

    choreo_time = mean(r["choreo_time_ms"] for r in rows)
    mlir_time = mean(r["mlir_time_ms"] for r in rows)
    render_bar_chart(
        plt,
        ["Choreo", "MLIR"],
        [choreo_time, mlir_time],
        "Average analysis time (ms)",
        "Average analysis/pipeline latency",
        output_dir / "analysis_latency_ms",
        ["#377eb8", "#e41a1c"],
    )

    summary_path = output_dir / "plot_summary.txt"
    summary_path.write_text(
        "Rendered figures:\n"
        f"- {output_dir / 'success_rate.png'}\n"
        f"- {output_dir / 'dynamic_symbolic_rate.png'}\n"
        f"- {output_dir / 'analysis_latency_ms.png'}\n"
    )


if __name__ == "__main__":
    main()
