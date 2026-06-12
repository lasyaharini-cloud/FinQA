from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "retrieval"
GEN_DIR = PROJECT_ROOT / "outputs" / "generation"
PLOT_DIR = PROJECT_ROOT / "plots" / "generation"
DATASET_FILE = PROJECT_ROOT / "FinQA" / "dataset" / "dev.json"
DETAILS_FILE = GEN_DIR / "program_evaluation_details.csv"
UNITS_FILE = PROJECT_ROOT / "data" / "processed" / "finqa_dev_evidence_units.csv"
RANKED_FILE = RETRIEVAL_DIR / "ranked_evidence_results.csv"

STEP_OP_RE = re.compile(r"[a-z_]+")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], y_label: str = "Value", as_percent: bool = False) -> None:
    width, height = 920, 520
    margin = 76
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_val = max([v for _, v in rows] + [1])
    if as_percent:
        max_val = 1.0
    bar_gap = 24
    bar_w = max(28, (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows)))
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        val = max_val * i / 5
        y = height - margin - (val / max_val) * plot_h
        label = f"{val:.0%}" if as_percent else f"{val:.0f}"
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{label}</text>')
    for i, (label, value) in enumerate(rows):
        x = margin + i * (bar_w + bar_gap)
        bar_h = (value / max_val) * plot_h if max_val else 0
        y = height - margin - bar_h
        color = colors[i % len(colors)]
        shown = f"{value:.1%}" if as_percent else f"{value:.0f}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{shown}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="12" font-family="Arial">{svg_escape(label)}</text>')
    parts.append(f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-size="13" font-family="Arial">{svg_escape(y_label)}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_chart(path: Path, title: str, groups: list[str], series: list[tuple[str, list[float]]], as_percent: bool = True) -> None:
    width, height = 980, 560
    margin = 80
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_val = 1.0 if as_percent else max([v for _, vals in series for v in vals] + [1])
    group_w = plot_w / max(1, len(groups))
    bar_w = group_w / (len(series) + 1.6)
    colors = ["#2563eb", "#d97706", "#16a34a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        val = max_val * i / 5
        y = height - margin - (val / max_val) * plot_h
        label = f"{val:.0%}" if as_percent else f"{val:.0f}"
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{label}</text>')
    for gi, group in enumerate(groups):
        base_x = margin + gi * group_w + group_w * 0.16
        for si, (_, values) in enumerate(series):
            value = values[gi]
            bar_h = (value / max_val) * plot_h if max_val else 0
            x = base_x + si * bar_w
            y = height - margin - bar_h
            label = f"{value:.0%}" if as_percent else f"{value:.0f}"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5:.1f}" height="{bar_h:.1f}" fill="{colors[si % len(colors)]}"/>')
            parts.append(f'<text x="{x + (bar_w-5)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{margin + gi * group_w + group_w/2:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="12" font-family="Arial">{svg_escape(group)}</text>')
    lx = width - margin - 230
    for si, (name, _) in enumerate(series):
        y = 76 + si * 24
        parts.append(f'<rect x="{lx}" y="{y-12}" width="14" height="14" fill="{colors[si % len(colors)]}"/>')
        parts.append(f'<text x="{lx+22}" y="{y}" font-size="13" font-family="Arial">{svg_escape(name)}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def make_retrieval_hit_rate_plot() -> None:
    rows = read_csv(RETRIEVAL_DIR / "topk_hit_rates.csv")
    modes = ["text_only", "table_only", "combined"]
    ks = ["1", "3", "5", "10"]
    by_key = {(row["mode"], row["k"]): float(row["hit_rate"]) for row in rows}
    series = []
    for mode in modes:
        series.append((mode, [by_key[(mode, k)] for k in ks]))
    groups = [f"hit@{k}" for k in ks]
    write_grouped_chart(
        PLOT_DIR / "retrieval_hit_rates_by_source.svg",
        "Retrieval Hit Rates by Evidence Source",
        groups,
        series,
    )

def make_operation_and_category_distribution_plots() -> None:
    rows = read_csv(DETAILS_FILE)
    op_counts = Counter(row["gold_operation"] for row in rows)
    cat_counts = Counter(row["question_category"] for row in rows)
    write_bar_chart(
        PLOT_DIR / "arithmetic_operation_mix.svg",
        "Arithmetic Program Operation Mix",
        sorted(op_counts.items()),
        y_label="Examples",
    )
    write_bar_chart(
        PLOT_DIR / "question_type_mix.svg",
        "Question Type Mix in Arithmetic Subset",
        sorted(cat_counts.items()),
        y_label="Examples",
    )


def make_source_coverage_plot() -> None:
    rows = read_csv(RETRIEVAL_DIR / "source_gold_coverage.csv")
    values = [(row["source_type"], float(row["coverage_rate"])) for row in rows]
    write_bar_chart(
        PLOT_DIR / "evidence_source_gold_coverage.svg",
        "Gold Evidence Coverage by Source",
        values,
        y_label="Coverage",
        as_percent=True,
    )


def main() -> None:
    make_retrieval_hit_rate_plot()
    make_operation_and_category_distribution_plots()
    make_source_coverage_plot()
    print(f"Saved extra analysis plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
