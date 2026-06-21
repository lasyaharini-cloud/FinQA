# FinQA Finance RAG Starter

This is a beginner-friendly starter for a finance-focused retrieval project with an audit angle using the public FinQA dataset.

## Project idea 1

Build a system that:

1. loads finance question-answer examples from FinQA,
2. inspects the text, tables, and gold evidence,
3. retrieves relevant evidence passages,
4. later extends to answer generation and simple support checking.

## Beginner-friendly scope

Start small:

- inspect the dataset structure,
- build a retrieval baseline,
- study whether the retrieved evidence looks relevant.

Do not try to build the full audit pipeline immediately.

## Dataset

- FinQA paper: https://arxiv.org/abs/2109.00122
- FinQA GitHub: https://github.com/czyssrs/FinQA

The official FinQA repository is cloned locally in:

- `/Users/lasya/finance_audit_rag/FinQA`

## Main packages

- `datasets`
- `pandas`
- `numpy`
- `sentence-transformers`
- `haystack-ai`
- `scikit-learn`
- `llama-index`
- `ragas`
- `jupyter`

## Open-source package we are adapting

We are using Haystack's local retrieval pipeline pattern and customizing it for FinanceBench.

- Haystack get started: https://docs.haystack.deepset.ai/docs/get_started
- InMemoryDocumentStore: https://docs.haystack.deepset.ai/docs/inmemorydocumentstore
- InMemoryEmbeddingRetriever: https://docs.haystack.deepset.ai/v2.8/docs/inmemoryembeddingretriever

Our custom change is that, instead of indexing generic demo sentences, we index FinQA text snippets and table rows with question metadata and gold evidence indicators.

## First commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/inspect_finqa.py
```

## First milestone

By the first milestone, you should be able to:

- load FinQA,
- inspect a few questions, text blocks, and table rows,
- save a preview CSV,
- explain your project direction clearly.

## Retrieval-to-Generation Evaluation

The project now has an end-to-end path from FinQA retrieval to program-generation evaluation:

```text
FinQA question
  -> retrieved evidence
  -> Qwen3 generated reasoning program
  -> program validation
  -> program execution
  -> numerical answer accuracy
```

### Retrieval outputs

Run the retrieval pipeline:

```bash
python3 scripts/finqa_retrieval.py all --split dev
```

Important outputs:

- `outputs/retrieval/topk_hit_rates.csv`
- `outputs/retrieval/ranked_evidence_results.csv`
- `plots/retrieval/text_table_combined_hit_at_k.svg`

### Qwen3 program generation

Generate FinQA-style arithmetic programs from retrieved evidence:

```bash
python3 scripts/generate_finqa_programs_qwen3.py \
  --input-csv outputs/finqa_topk_results_same_example.csv \
  --output-csv outputs/finqa_qwen3_programs.csv \
  --max-examples 25
```

This writes generated programs such as:

```text
divide(24800, 15400)
```

### Program evaluation

Evaluate real Qwen outputs:

```bash
python3 scripts/evaluate_finqa_programs.py \
  --predictions outputs/finqa_qwen3_programs.csv \
  --arithmetic-only
```

If Qwen has not been run yet, use the gold-program smoke test to verify that the evaluator, executor, CSVs, and plots work:

```bash
python3 scripts/evaluate_finqa_programs.py \
  --use-gold-programs \
  --arithmetic-only \
  --max-examples 150
```

Smoke-test outputs are intentionally an oracle check, not Qwen performance.

Evaluation outputs:

- `outputs/generation/program_evaluation_details.csv`
- `outputs/generation/program_evaluation_by_source.csv`
- `outputs/generation/program_evaluation_by_operation.csv`
- `outputs/generation/program_evaluation_by_category.csv`
- `outputs/generation/program_evaluation_summary.md`

PPT-ready plots:

- `plots/generation/program_metrics_by_source.svg`
- `plots/generation/answer_accuracy_by_operation.svg`
- `plots/generation/answer_accuracy_by_category.svg`

Metrics:

- Program validity rate: generated program uses allowed FinQA operations and valid arguments.
- Program exact match: generated program exactly matches the canonical gold program.
- Execution rate: generated program can be executed without errors.
- Answer accuracy: executed numerical result matches the FinQA gold answer within tolerance.

