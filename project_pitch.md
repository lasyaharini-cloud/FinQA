# Project Pitch

I am interested in working on a retrieval-augmented generation (RAG) project for financial question answering using the FinQA dataset. The idea is to build a system where someone asks a question about a financial report, the system searches the relevant text and table content, finds the most relevant parts, and then uses those parts to support the answer.

I would like to begin with a simple baseline retrieval system first. By that, I mean starting with a basic model that only focuses on identifying and ranking the most relevant passages or table rows for a financial question, before moving to answer generation or more advanced evidence checking.

This project is especially interesting to me because of my previous background in internal audit, where conclusions needed to be supported by documentation. I would like to bring that perspective into an LLM/RAG project by studying whether answers are actually supported by the retrieved evidence.

For the technical side, I have been looking at Sentence Transformers for retrieval embeddings and Haystack as a retrieval pipeline framework.

The dataset resources I have been looking at are:

- FinQA paper: https://arxiv.org/abs/2109.00122
- FinQA GitHub: https://github.com/czyssrs/FinQA

