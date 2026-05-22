from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR = Path("outputs")
UNITS_CSV = OUTPUT_DIR / "finqa_retrieval_units.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a baseline FinQA retriever over retrieval-ready units."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="How many unique questions/examples to evaluate first.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieval results to keep per question.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["global_subset", "same_example"],
        default="global_subset",
        help=(
            "global_subset ranks against all units in the selected subset; "
            "same_example ranks only within the current example."
        ),
    )
    return parser.parse_args()


def build_global_subset(df: pd.DataFrame, num_questions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    question_df = (
        df[["example_id", "filename", "question", "answer", "exe_ans"]]
        .drop_duplicates(subset=["example_id"])
        .head(num_questions)
        .reset_index(drop=True)
    )
    example_ids = set(question_df["example_id"])
    candidate_df = df[df["example_id"].isin(example_ids)].reset_index(drop=True)
    return question_df, candidate_df


def retrieve_with_global_subset(
    model: SentenceTransformer, question_df: pd.DataFrame, candidate_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    question_embeddings = model.encode(
        question_df["question"].tolist(), convert_to_numpy=True, show_progress_bar=True
    )
    candidate_embeddings = model.encode(
        candidate_df["unit_text"].tolist(), convert_to_numpy=True, show_progress_bar=True
    )

    sim = cosine_similarity(question_embeddings, candidate_embeddings)
    rows_out: list[dict] = []

    for q_idx, q_row in question_df.iterrows():
        top_idx = sim[q_idx].argsort()[-top_k:][::-1]
        for rank, cand_idx in enumerate(top_idx, start=1):
            cand = candidate_df.iloc[cand_idx]
            rows_out.append(
                {
                    "query_example_id": q_row["example_id"],
                    "query_filename": q_row["filename"],
                    "query": q_row["question"],
                    "gold_answer": q_row["answer"],
                    "candidate_scope": "global_subset",
                    "retrieved_example_id": cand["example_id"],
                    "retrieved_filename": cand["filename"],
                    "unit_id": cand["unit_id"],
                    "unit_type": cand["unit_type"],
                    "unit_index": cand["unit_index"],
                    "retrieved_text": cand["unit_text"],
                    "score": float(sim[q_idx, cand_idx]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows_out)


def retrieve_with_same_example(
    model: SentenceTransformer, question_df: pd.DataFrame, candidate_df: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    rows_out: list[dict] = []
    for _, q_row in question_df.iterrows():
        local_df = candidate_df[candidate_df["example_id"] == q_row["example_id"]].reset_index(drop=True)
        question_embedding = model.encode([q_row["question"]], convert_to_numpy=True, show_progress_bar=False)
        local_embeddings = model.encode(
            local_df["unit_text"].tolist(), convert_to_numpy=True, show_progress_bar=False
        )
        sim = cosine_similarity(question_embedding, local_embeddings)[0]
        top_idx = sim.argsort()[-top_k:][::-1]

        for rank, cand_idx in enumerate(top_idx, start=1):
            cand = local_df.iloc[cand_idx]
            rows_out.append(
                {
                    "query_example_id": q_row["example_id"],
                    "query_filename": q_row["filename"],
                    "query": q_row["question"],
                    "gold_answer": q_row["answer"],
                    "candidate_scope": "same_example",
                    "retrieved_example_id": cand["example_id"],
                    "retrieved_filename": cand["filename"],
                    "unit_id": cand["unit_id"],
                    "unit_type": cand["unit_type"],
                    "unit_index": cand["unit_index"],
                    "retrieved_text": cand["unit_text"],
                    "score": float(sim[cand_idx]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows_out)


def main() -> None:
    args = parse_args()
    print("Loading retrieval-ready FinQA units...")
    df = pd.read_csv(UNITS_CSV)

    question_df, candidate_df = build_global_subset(df, args.num_questions)
    print(f"Questions selected: {len(question_df)}")
    print(f"Candidate units in subset: {len(candidate_df)}")
    print(f"Candidate scope: {args.candidate_scope}")
    print(f"Top-k: {args.top_k}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    if args.candidate_scope == "global_subset":
        results = retrieve_with_global_subset(model, question_df, candidate_df, args.top_k)
    else:
        results = retrieve_with_same_example(model, question_df, candidate_df, args.top_k)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_file = OUTPUT_DIR / f"finqa_topk_results_{args.candidate_scope}.csv"
    results.to_csv(out_file, index=False)

    print()
    print("Sample retrieval results:")
    sample_cols = ["query", "unit_type", "score", "rank"]
    print(results[sample_cols].head(10).to_string(index=False))
    print()
    print(f"Saved retrieval results to {out_file.resolve()}")


if __name__ == "__main__":
    main()
