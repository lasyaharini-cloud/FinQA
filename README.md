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
