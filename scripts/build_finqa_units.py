from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("FinQA/dataset")
OUTPUT_DIR = Path("outputs")


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())


def table_row_to_text(row: list[str]) -> str:
    cleaned = [str(cell).strip() for cell in row if str(cell).strip()]
    return " | ".join(cleaned)


def build_units_for_example(example: dict) -> list[dict]:
    units: list[dict] = []
    qa = example["qa"]
    base_meta = {
        "example_id": example["id"],
        "filename": example["filename"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "exe_ans": qa.get("exe_ans"),
    }

    for idx, text in enumerate(example.get("pre_text", [])):
        text = text.strip()
        if not text:
            continue
        units.append(
            {
                **base_meta,
                "unit_id": f"{example['id']}_pre_{idx}",
                "unit_type": "pre_text",
                "unit_index": idx,
                "unit_text": text,
            }
        )

    for idx, text in enumerate(example.get("post_text", [])):
        text = text.strip()
        if not text:
            continue
        units.append(
            {
                **base_meta,
                "unit_id": f"{example['id']}_post_{idx}",
                "unit_type": "post_text",
                "unit_index": idx,
                "unit_text": text,
            }
        )

    for idx, row in enumerate(example.get("table", [])):
        row_text = table_row_to_text(row)
        if not row_text:
            continue
        units.append(
            {
                **base_meta,
                "unit_id": f"{example['id']}_table_{idx}",
                "unit_type": "table",
                "unit_index": idx,
                "unit_text": row_text,
            }
        )

    return units


def main() -> None:
    print("Loading FinQA train split...")
    train = load_split("train")

    print("Building retrieval-ready units from pre_text, post_text, and table rows...")
    all_units: list[dict] = []
    for example in train:
        all_units.extend(build_units_for_example(example))

    df = pd.DataFrame(all_units)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = OUTPUT_DIR / "finqa_retrieval_units.csv"
    df.to_csv(out_csv, index=False)

    print(f"Train examples processed: {len(train)}")
    print(f"Total retrieval units created: {len(df)}")
    print()
    print("Unit counts by type:")
    for unit_type, count in df["unit_type"].value_counts().items():
        print(f"- {unit_type}: {count}")
    print()
    print(f"Saved retrieval units to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
