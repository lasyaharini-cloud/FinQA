from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrieval"
PLOT_DIR = PROJECT_ROOT / "plots" / "retrieval"
LOCAL_FINQA_DIR = PROJECT_ROOT / "FinQA" / "dataset"

FINQA_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/train.json",
    "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/dev.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json",
}


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")


@dataclass(frozen=True)
class EvidenceUnit:
    example_id: str
    question: str
    answer: str
    unit_id: str
    source_type: str
    evidence_text: str
    is_gold: bool
    gold_key: str


def ensure_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_finqa(force: bool = False) -> None:
    ensure_dirs()
    for split, url in FINQA_URLS.items():
        out_file = RAW_DIR / f"{split}.json"
        if out_file.exists() and not force:
            print(f"Already have {out_file}")
            continue
        local_file = LOCAL_FINQA_DIR / f"{split}.json"
        if local_file.exists() and not force:
            shutil.copyfile(local_file, out_file)
            print(f"Copied local FinQA {split} split from {local_file}")
            continue
        print(f"Downloading {split} from FinQA GitHub...")
        with urllib.request.urlopen(url, timeout=60) as response:
            out_file.write_bytes(response.read())
        print(f"Saved {out_file}")


def load_split(split: str) -> list[dict]:
    file_path = RAW_DIR / f"{split}.json"
    local_finqa_file = PROJECT_ROOT / "FinQA" / "dataset" / f"{split}.json"
    if not file_path.exists() and local_finqa_file.exists():
        file_path = local_finqa_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {file_path}. Run: python scripts/finqa_retrieval.py download"
        )
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
        raise ValueError(f"Could not find a list of examples in {file_path}")
    return data


def normalize_space(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def table_row_to_text(row: object, row_index: int) -> str:
    if isinstance(row, list):
        cells = [normalize_space(cell) for cell in row]
        return f"table row {row_index}: " + " | ".join(cells)
    if isinstance(row, dict):
        cells = [f"{normalize_space(k)}: {normalize_space(v)}" for k, v in row.items()]
        return f"table row {row_index}: " + " | ".join(cells)
    return f"table row {row_index}: {normalize_space(row)}"


def get_question(example: dict) -> str:
    qa = example.get("qa") or {}
    return normalize_space(qa.get("question") or example.get("question"))


def get_answer(example: dict) -> str:
    qa = example.get("qa") or {}
    return normalize_space(qa.get("answer") or qa.get("exe_ans") or example.get("answer"))


def get_gold_keys(example: dict) -> set[str]:
    qa = example.get("qa") or {}
    raw_gold = qa.get("gold_inds") or qa.get("gold_ind") or example.get("gold_inds") or {}
    if isinstance(raw_gold, dict):
        return {normalize_space(key) for key in raw_gold.keys()}
    if isinstance(raw_gold, list):
        return {normalize_space(item) for item in raw_gold}
    return set()


def text_overlap_score(query: str, text: str) -> float:
    query_tokens = {stemish(token) for token in tokenize(query)}
    text_tokens = {stemish(token) for token in tokenize(text)}
    if not query_tokens or not text_tokens:
        return 0.0
    lexical = len(query_tokens & text_tokens) / len(query_tokens | text_tokens)
    numeric = len(numbers(query) & numbers(text)) * 0.25
    return lexical + numeric


def select_context_text(
    question: str,
    table_text: str,
    text_rows: list[tuple[str, str]],
    max_rows: int = 3,
) -> list[tuple[str, str]]:
    scored = []
    query = f"{question} {table_text}"
    for key, text in text_rows:
        scored.append((text_overlap_score(query, text), key, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(key, text) for score, key, text in scored[:max_rows] if score > 0]


def iter_units(example: dict, example_index: int) -> Iterable[EvidenceUnit]:
    example_id = normalize_space(example.get("id") or example.get("filename") or example_index)
    question = get_question(example)
    answer = get_answer(example)
    gold_keys = get_gold_keys(example)

    pre_text = example.get("pre_text") or []
    post_text = example.get("post_text") or []
    table = example.get("table") or []

    text_rows: list[tuple[str, str]] = []
    table_rows: list[tuple[str, str]] = []

    for i, sentence in enumerate(pre_text):
        key = f"text_{i}"
        text = normalize_space(sentence)
        if text:
            text_rows.append((key, text))
            yield EvidenceUnit(
                example_id, question, answer, key, "text", text, key in gold_keys, key
            )

    for i, row in enumerate(table):
        key = f"table_{i}"
        text = table_row_to_text(row, i)
        if text:
            table_rows.append((key, text))
            yield EvidenceUnit(
                example_id, question, answer, key, "table", text, key in gold_keys, key
            )

    for i, sentence in enumerate(post_text):
        key = f"text_{len(pre_text) + i}"
        text = normalize_space(sentence)
        if text:
            text_rows.append((key, text))
            yield EvidenceUnit(
                example_id, question, answer, key, "text", text, key in gold_keys, key
            )

    if text_rows and table_rows:
        for key, row_text in table_rows:
            combined_key = f"combined_{key}"
            context_rows = select_context_text(question, row_text, text_rows)
            context_text = " ".join(text for _, text in context_rows)
            combined_text = f"{row_text} [matched text context] {context_text}"
            context_keys = {text_key for text_key, _ in context_rows}
            is_gold = key in gold_keys or bool(context_keys & gold_keys)
            yield EvidenceUnit(
                example_id,
                question,
                answer,
                combined_key,
                "combined",
                combined_text,
                is_gold,
                key if key in gold_keys else ",".join(sorted(context_keys & gold_keys)),
            )


def prepare_units(split: str = "dev", limit: int | None = None) -> list[EvidenceUnit]:
    ensure_dirs()
    examples = load_split(split)
    if limit is not None:
        examples = examples[:limit]

    units: list[EvidenceUnit] = []
    skipped = 0
    for i, example in enumerate(examples):
        question = get_question(example)
        gold_keys = get_gold_keys(example)
        if not question or not gold_keys:
            skipped += 1
            continue
        units.extend(iter_units(example, i))

    out_file = PROCESSED_DIR / f"finqa_{split}_evidence_units.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "question",
                "answer",
                "unit_id",
                "source_type",
                "evidence_text",
                "is_gold",
                "gold_key",
            ],
        )
        writer.writeheader()
        for unit in units:
            writer.writerow(unit.__dict__)

    examples_with_gold = len({unit.example_id for unit in units if unit.is_gold})
    print(f"Prepared {len(units):,} evidence units from {split}")
    print(f"Examples with at least one gold unit: {examples_with_gold:,}")
    print(f"Skipped examples without question/gold labels: {skipped:,}")
    print(f"Saved {out_file}")
    return units


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def numbers(text: str) -> set[str]:
    cleaned = set()
    for match in NUMBER_RE.findall(text):
        cleaned.add(match.replace(",", "").rstrip("%"))
    return cleaned


def stemish(token: str) -> str:
    for suffix in ["ing", "tion", "ions", "ed", "ly", "es", "s"]:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class EvidenceRetriever:
    """Small modern retriever: BM25 + character overlap + numeric-aware reranking."""

    def __init__(self, units: list[EvidenceUnit]):
        self.units = units
        self.docs = [tokenize(unit.evidence_text) for unit in units]
        self.stem_docs = [[stemish(token) for token in doc] for doc in self.docs]
        self.doc_numbers = [numbers(unit.evidence_text) for unit in units]
        self.doc_freq: Counter[str] = Counter()
        for doc in self.stem_docs:
            self.doc_freq.update(set(doc))
        self.avgdl = statistics.mean(len(doc) for doc in self.stem_docs) if self.stem_docs else 1

    def bm25(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        n_docs = len(self.stem_docs)
        score = 0.0
        k1 = 1.4
        b = 0.72
        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / self.avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    @staticmethod
    def char_jaccard(a: str, b: str) -> float:
        def grams(text: str) -> set[str]:
            clean = re.sub(r"[^a-z0-9]+", " ", text.lower())
            compact = clean.replace(" ", "_")
            return {compact[i : i + 4] for i in range(max(0, len(compact) - 3))}

        a_grams = grams(a)
        b_grams = grams(b)
        if not a_grams or not b_grams:
            return 0.0
        return len(a_grams & b_grams) / len(a_grams | b_grams)

    def search(self, question: str, top_k: int = 10) -> list[tuple[EvidenceUnit, float]]:
        q_tokens = [stemish(token) for token in tokenize(question)]
        q_numbers = numbers(question)
        scored: list[tuple[EvidenceUnit, float]] = []
        for idx, unit in enumerate(self.units):
            bm25_score = self.bm25(q_tokens, self.stem_docs[idx])
            char_score = self.char_jaccard(question, unit.evidence_text)
            numeric_bonus = len(q_numbers & self.doc_numbers[idx]) * 0.75
            source_prior = 0.15 if unit.source_type == "combined" else 0.0
            score = bm25_score + (2.0 * char_score) + numeric_bonus + source_prior
            scored.append((unit, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def load_units(split: str) -> list[EvidenceUnit]:
    file_path = PROCESSED_DIR / f"finqa_{split}_evidence_units.csv"
    if not file_path.exists():
        return prepare_units(split)
    units = []
    with file_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            units.append(
                EvidenceUnit(
                    example_id=row["example_id"],
                    question=row["question"],
                    answer=row["answer"],
                    unit_id=row["unit_id"],
                    source_type=row["source_type"],
                    evidence_text=row["evidence_text"],
                    is_gold=str(row["is_gold"]).lower() == "true",
                    gold_key=row["gold_key"],
                )
            )
    return units


def evaluate(split: str = "dev", max_examples: int | None = None, top_k: int = 10) -> None:
    ensure_dirs()
    units = load_units(split)
    by_example: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for unit in units:
        by_example[unit.example_id].append(unit)

    example_ids = sorted(by_example.keys())
    if max_examples is not None:
        example_ids = example_ids[:max_examples]

    modes = {
        "text_only": {"text"},
        "table_only": {"table"},
        "combined": {"combined"},
    }
    ks = [1, 3, 5, 10]
    metric_rows = []
    example_rows = []
    first_rank_rows = []
    coverage_rows = []

    all_evaluated_examples = {
        example_id
        for example_id in example_ids
        if any(unit.is_gold for unit in by_example[example_id])
    }
    for source_type in ["text", "table", "combined"]:
        examples_with_source_gold = {
            example_id
            for example_id in example_ids
            if any(
                unit.is_gold and unit.source_type == source_type
                for unit in by_example[example_id]
            )
        }
        coverage_rows.append(
            {
                "source_type": source_type,
                "examples_with_gold_source": len(examples_with_source_gold),
                "all_gold_examples": len(all_evaluated_examples),
                "coverage_rate": round(
                    len(examples_with_source_gold) / len(all_evaluated_examples), 4
                )
                if all_evaluated_examples
                else 0.0,
            }
        )

    for mode, allowed_sources in modes.items():
        hits = {k: 0 for k in ks}
        evaluated = 0
        first_ranks: list[int] = []

        for example_id in example_ids:
            example_units = [u for u in by_example[example_id] if u.source_type in allowed_sources]
            if not example_units:
                continue
            if not any(unit.is_gold for unit in example_units):
                continue

            question = example_units[0].question
            retriever = EvidenceRetriever(example_units)
            ranked = retriever.search(question, top_k=top_k)
            evaluated += 1
            first_gold_rank = None

            for rank, (unit, score) in enumerate(ranked, start=1):
                example_rows.append(
                    {
                        "mode": mode,
                        "example_id": example_id,
                        "question": question,
                        "answer": unit.answer,
                        "rank": rank,
                        "score": round(score, 6),
                        "source_type": unit.source_type,
                        "unit_id": unit.unit_id,
                        "is_gold": unit.is_gold,
                        "evidence_text": unit.evidence_text,
                    }
                )
                if unit.is_gold and first_gold_rank is None:
                    first_gold_rank = rank

            if first_gold_rank is not None:
                first_ranks.append(first_gold_rank)
                for k in ks:
                    if first_gold_rank <= k:
                        hits[k] += 1

        for k in ks:
            metric_rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "evaluated_examples": evaluated,
                    "hits": hits[k],
                    "hit_rate": round(hits[k] / evaluated, 4) if evaluated else 0.0,
                }
            )
        for rank in first_ranks:
            first_rank_rows.append({"mode": mode, "first_gold_rank": rank})

    metrics_file = OUTPUT_DIR / "topk_hit_rates.csv"
    examples_file = OUTPUT_DIR / "ranked_evidence_results.csv"
    ranks_file = OUTPUT_DIR / "first_gold_ranks.csv"
    coverage_file = OUTPUT_DIR / "source_gold_coverage.csv"
    write_csv(metrics_file, metric_rows)
    write_csv(examples_file, example_rows)
    write_csv(ranks_file, first_rank_rows)
    write_csv(coverage_file, coverage_rows)
    plot_hit_rates(metric_rows, PLOT_DIR / "text_table_combined_hit_at_k.svg")
    plot_first_ranks(first_rank_rows, PLOT_DIR / "first_gold_rank_distribution.svg")
    plot_source_coverage(coverage_rows, PLOT_DIR / "source_gold_coverage.svg")
    write_summary(metric_rows, first_rank_rows, coverage_rows)
    write_demo_notes(metric_rows, coverage_rows)

    print(f"Saved metrics: {metrics_file}")
    print(f"Saved ranked examples: {examples_file}")
    print(f"Saved first-rank data: {ranks_file}")
    print(f"Saved source coverage: {coverage_file}")
    print(f"Saved plots in {PLOT_DIR}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def plot_hit_rates(rows: list[dict], path: Path) -> None:
    modes = ["text_only", "table_only", "combined"]
    colors = {
        "text_only": "#2563eb",
        "table_only": "#16a34a",
        "combined": "#d97706",
    }
    by_key = {(row["mode"], int(row["k"])): float(row["hit_rate"]) for row in rows}
    ks = [1, 3, 5, 10]
    width, height = 860, 520
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_y = max([float(row["hit_rate"]) for row in rows] + [0.1])
    max_y = min(1.0, max(0.2, math.ceil(max_y * 10) / 10))
    x_gap = plot_w / (len(ks) - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="430" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">FinQA Evidence Retrieval: Hit@K by Source</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        y_val = max_y * i / 5
        y = height - margin - (y_val / max_y) * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.2f}</text>')
    for i, k in enumerate(ks):
        x = margin + i * x_gap
        parts.append(f'<text x="{x:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="13" font-family="Arial">hit@{k}</text>')
    for mode in modes:
        points = []
        for i, k in enumerate(ks):
            x = margin + i * x_gap
            y = height - margin - (by_key.get((mode, k), 0.0) / max_y) * plot_h
            points.append((x, y))
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{colors[mode]}" stroke-width="3" points="{point_str}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[mode]}"/>')
    for i, mode in enumerate(modes):
        y = 86 + i * 24
        parts.append(f'<rect x="650" y="{y-12}" width="14" height="14" fill="{colors[mode]}"/>')
        parts.append(f'<text x="672" y="{y}" font-size="13" font-family="Arial">{svg_escape(mode)}</text>')
    parts.append('<text x="430" y="502" text-anchor="middle" font-size="13" font-family="Arial">Top-k evidence list size</text>')
    parts.append('<text x="18" y="260" transform="rotate(-90 18 260)" text-anchor="middle" font-size="13" font-family="Arial">Hit rate</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_first_ranks(rows: list[dict], path: Path) -> None:
    modes = ["text_only", "table_only", "combined"]
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]
    buckets = ["1", "2-3", "4-5", "6-10"]

    def bucket(rank: int) -> str | None:
        if rank == 1:
            return "1"
        if 2 <= rank <= 3:
            return "2-3"
        if 4 <= rank <= 5:
            return "4-5"
        if 6 <= rank <= 10:
            return "6-10"
        return None

    counts = {(mode, b): 0 for mode in modes for b in buckets}
    for row in rows:
        b = bucket(int(row["first_gold_rank"]))
        if b:
            counts[(row["mode"], b)] += 1

    width, height = 900, 520
    margin = 72
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_count = max(counts.values() or [1])
    group_w = plot_w / len(buckets)
    bar_w = group_w / 6

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">First Correct Evidence Rank</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        y_val = max_count * i / 5
        y = height - margin - (y_val / max_count) * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.0f}</text>')
    for bi, b in enumerate(buckets):
        x_center = margin + bi * group_w + group_w / 2
        parts.append(f'<text x="{x_center:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="13" font-family="Arial">rank {b}</text>')
        for mi, mode in enumerate(modes):
            value = counts[(mode, b)]
            bar_h = (value / max_count) * plot_h
            x = margin + bi * group_w + group_w * 0.18 + mi * bar_w
            y = height - margin - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" height="{bar_h:.1f}" fill="{colors[mi]}"/>')
    for i, mode in enumerate(modes):
        y = 86 + i * 24
        parts.append(f'<rect x="680" y="{y-12}" width="14" height="14" fill="{colors[i]}"/>')
        parts.append(f'<text x="702" y="{y}" font-size="13" font-family="Arial">{svg_escape(mode)}</text>')
    parts.append('<text x="450" y="502" text-anchor="middle" font-size="13" font-family="Arial">First gold evidence rank bucket</text>')
    parts.append('<text x="18" y="260" transform="rotate(-90 18 260)" text-anchor="middle" font-size="13" font-family="Arial">Number of examples</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_source_coverage(rows: list[dict], path: Path) -> None:
    colors = {"text": "#2563eb", "table": "#16a34a", "combined": "#d97706"}
    width, height = 780, 420
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = plot_w / 5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="390" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">Gold Evidence Coverage by Source</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        y_val = i / 5
        y = height - margin - y_val * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.1f}</text>')
    for i, row in enumerate(rows):
        source = row["source_type"]
        value = float(row["coverage_rate"])
        x = margin + 80 + i * (bar_w + 70)
        bar_h = value * plot_h
        y = height - margin - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[source]}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="13" font-family="Arial">{value:.1%}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="13" font-family="Arial">{source}</text>')
    parts.append('<text x="390" y="400" text-anchor="middle" font-size="13" font-family="Arial">Evidence source</text>')
    parts.append('<text x="18" y="210" transform="rotate(-90 18 210)" text-anchor="middle" font-size="13" font-family="Arial">Share of dev examples with gold evidence</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(
    metric_rows: list[dict],
    first_rank_rows: list[dict],
    coverage_rows: list[dict],
) -> None:
    best_by_k = {}
    for k in [1, 3, 5, 10]:
        rows = [row for row in metric_rows if int(row["k"]) == k]
        best_by_k[k] = max(rows, key=lambda row: float(row["hit_rate"]))

    ranks_by_mode: dict[str, list[int]] = defaultdict(list)
    for row in first_rank_rows:
        ranks_by_mode[row["mode"]].append(int(row["first_gold_rank"]))

    lines = [
        "# FinQA Retrieval Summary",
        "",
        "This run evaluates evidence retrieval only. It does not reproduce the full FinQA program-generation benchmark.",
        "",
        "## Best Source by K",
    ]
    for k, row in best_by_k.items():
        lines.append(
            f"- hit@{k}: {row['mode']} = {float(row['hit_rate']):.1%} "
            f"({row['hits']}/{row['evaluated_examples']} examples)"
        )
    lines.extend(["", "## Median First Correct Rank"])
    for mode, ranks in sorted(ranks_by_mode.items()):
        if ranks:
            lines.append(f"- {mode}: {statistics.median(ranks):.1f}")
    lines.extend(["", "## Gold Evidence Coverage"])
    for row in coverage_rows:
        lines.append(
            f"- {row['source_type']}: {float(row['coverage_rate']):.1%} "
            f"({row['examples_with_gold_source']}/{row['all_gold_examples']} examples)"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- data/processed/finqa_dev_evidence_units.csv",
            "- outputs/retrieval/topk_hit_rates.csv",
            "- outputs/retrieval/ranked_evidence_results.csv",
            "- outputs/retrieval/source_gold_coverage.csv",
            "- plots/retrieval/text_table_combined_hit_at_k.svg",
            "- plots/retrieval/first_gold_rank_distribution.svg",
            "- plots/retrieval/source_gold_coverage.svg",
        ]
    )
    (OUTPUT_DIR / "retrieval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_demo_notes(metric_rows: list[dict], coverage_rows: list[dict]) -> None:
    by_key = {(row["mode"], int(row["k"])): row for row in metric_rows}
    lines = [
        "# Retrieval Demo Notes",
        "",
        "## One-sentence project story",
        "I converted each FinQA question into small searchable evidence units, then tested whether a newer retrieval scorer can find the gold evidence from text, tables, or contextual text+table bundles.",
        "",
        "## What to show first",
        "1. Show `data/processed/finqa_dev_evidence_units.csv`: this is the dataset after we break each example into searchable chunks.",
        "2. Show `outputs/retrieval/ranked_evidence_results.csv`: this is the ranked evidence list for each question.",
        "3. Show `outputs/retrieval/topk_hit_rates.csv`: this is where hit@1, hit@3, hit@5, and hit@10 are computed.",
        "4. Show the three SVG plots in `plots/retrieval/`.",
        "",
        "## Easy explanation",
        "Think of each question as asking: where is the proof? The retriever makes a top-10 list of possible proof pieces. If the real proof is in the list, we count it as a hit.",
        "",
        "## Result highlights",
    ]
    for mode in ["text_only", "table_only", "combined"]:
        row1 = by_key.get((mode, 1))
        row10 = by_key.get((mode, 10))
        if row1 and row10:
            lines.append(
                f"- {mode}: hit@1 = {float(row1['hit_rate']):.1%}, "
                f"hit@10 = {float(row10['hit_rate']):.1%} "
                f"on {row1['evaluated_examples']} evaluated examples."
            )
    lines.extend(["", "## Important note about denominators"])
    for row in coverage_rows:
        lines.append(
            f"- {row['source_type']} gold evidence exists in "
            f"{row['examples_with_gold_source']} of {row['all_gold_examples']} examples."
        )
    lines.extend(
        [
            "",
            "So text-only, table-only, and combined are evaluated on the examples where that source has gold evidence. This is fairer than forcing text to answer table-only questions.",
            "",
            "## Why this is not just the original FinQA paper",
            "The original paper is mainly about retrieving evidence and then generating executable programs for numerical reasoning. This project isolates retrieval quality and uses a lightweight modern retrieval scorer: BM25 lexical matching, fuzzy character overlap, numeric-aware scoring, and contextual text+table bundles.",
        ]
    )
    (OUTPUT_DIR / "demo_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinQA evidence retrieval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--force", action="store_true")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    prepare_parser.add_argument("--limit", type=int)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    run_parser.add_argument("--max-examples", type=int)
    run_parser.add_argument("--top-k", type=int, default=10)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    all_parser.add_argument("--max-examples", type=int)
    all_parser.add_argument("--force-download", action="store_true")

    args = parser.parse_args()
    if args.command == "download":
        download_finqa(force=args.force)
    elif args.command == "prepare":
        prepare_units(split=args.split, limit=args.limit)
    elif args.command == "run":
        evaluate(split=args.split, max_examples=args.max_examples, top_k=args.top_k)
    elif args.command == "all":
        download_finqa(force=args.force_download)
        prepare_units(split=args.split, limit=args.max_examples)
        evaluate(split=args.split, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
