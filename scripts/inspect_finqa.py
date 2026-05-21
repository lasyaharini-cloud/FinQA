from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("FinQA/dataset")


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())


def flatten_example(row: dict) -> dict:
    qa = row["qa"]
    return {
        "id": row["id"],
        "filename": row["filename"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "exe_ans": qa.get("exe_ans"),
        "program": " | ".join(qa.get("program", [])),
        "num_pre_text_blocks": len(row.get("pre_text", [])),
        "num_post_text_blocks": len(row.get("post_text", [])),
        "num_table_rows": len(row.get("table", [])),
        "num_gold_inds": len(qa.get("gold_inds", {})),
    }


def main() -> None:
    print("Loading FinQA directly from the official GitHub repository clone...")
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")

    print(f"Train examples: {len(train)}")
    print(f"Dev examples:   {len(dev)}")
    print(f"Test examples:  {len(test)}")
    print()

    first = train[0]
    qa = first["qa"]

    print("How one FinQA example is structured:")
    print("- pre_text: text before the table")
    print("- post_text: text after the table")
    print("- table: table rows from the report")
    print("- qa: question, answer, program, and gold evidence")
    print()

    print("First training question:")
    print(qa["question"])
    print()
    print("Reference answer:")
    print(qa.get("answer"))
    print()
    print("Executable answer:")
    print(qa.get("exe_ans"))
    print()
    print("Gold evidence keys:")
    print(list(qa.get("gold_inds", {}).keys())[:5])
    print()
    print("First 2 pre_text blocks:")
    for text in first.get("pre_text", [])[:2]:
        print(f"- {text}")
    print()
    print("First 3 table rows:")
    for row in first.get("table", [])[:3]:
        print(row)
    print()

    preview = pd.DataFrame(flatten_example(row) for row in train[:10])
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "finqa_preview.csv"
    preview.to_csv(out_file, index=False)
    print(f"Saved preview table to {out_file.resolve()}")


if __name__ == "__main__":
    main()
