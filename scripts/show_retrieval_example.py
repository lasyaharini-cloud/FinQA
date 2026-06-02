from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKED_FILE = PROJECT_ROOT / "outputs" / "retrieval" / "ranked_evidence_results.csv"
UNITS_FILE = PROJECT_ROOT / "data" / "processed" / "finqa_dev_evidence_units.csv"

MODES = ["text_only", "table_only", "combined"]
MODE_SOURCES = {
    "text_only": {"text"},
    "table_only": {"table"},
    "combined": {"combined"},
}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_example_with_combined_hit(rows: list[dict]) -> str:
    for row in rows:
        if row["mode"] == "combined" and row["rank"] == "1" and row["is_gold"] == "True":
            return row["example_id"]
    raise RuntimeError("No combined rank-1 hit found.")


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ranked FinQA retrieval evidence for one example.")
    parser.add_argument("--example-id", help="Example id to inspect. If omitted, uses a combined rank-1 hit.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chars", type=int, default=500)
    args = parser.parse_args()

    ranked_rows = load_csv(RANKED_FILE)
    unit_rows = load_csv(UNITS_FILE)
    example_id = args.example_id or first_example_with_combined_hit(ranked_rows)

    question = next((r["question"] for r in ranked_rows if r["example_id"] == example_id), "")
    answer = next((r["answer"] for r in ranked_rows if r["example_id"] == example_id), "")

    gold_by_mode: dict[str, set[str]] = defaultdict(set)
    for unit in unit_rows:
        if unit["example_id"] != example_id or unit["is_gold"].lower() != "true":
            continue
        for mode, sources in MODE_SOURCES.items():
            if unit["source_type"] in sources:
                gold_by_mode[mode].add(unit["unit_id"])

    print(f"EXAMPLE_ID: {example_id}")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print()

    for mode in MODES:
        rows = [
            r for r in ranked_rows
            if r["example_id"] == example_id and r["mode"] == mode and int(r["rank"]) <= args.top_k
        ]
        top_units = {r["unit_id"] for r in rows}
        gold_units = gold_by_mode[mode]
        any_hit = bool(gold_units & top_units)
        all_gold = bool(gold_units) and gold_units <= top_units

        print("=" * 72)
        print(f"MODE: {mode}")
        print(f"Gold units for this mode: {len(gold_units)}")
        print(f"Any gold in top {args.top_k}? {any_hit}")
        print(f"All gold covered in top {args.top_k}? {all_gold}")
        print()

        if not rows:
            print("No ranked candidates for this mode/example.")
            print()
            continue

        for row in rows:
            print(
                f"rank {row['rank']} | score {row['score']} | "
                f"gold? {row['is_gold']} | unit {row['unit_id']}"
            )
            print(shorten(row["evidence_text"], args.chars))
            print()


if __name__ == "__main__":
    main()
