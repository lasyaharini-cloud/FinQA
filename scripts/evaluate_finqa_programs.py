from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "FinQA" / "dataset" / "dev.json"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "finqa_qwen3_programs.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generation"
PLOT_DIR = PROJECT_ROOT / "plots" / "generation"

ALLOWED_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
CONST_VALUES = {
    "const_1": 1.0,
    "const_2": 2.0,
    "const_3": 3.0,
    "const_4": 4.0,
    "const_5": 5.0,
    "const_6": 6.0,
    "const_7": 7.0,
    "const_8": 8.0,
    "const_9": 9.0,
    "const_10": 10.0,
    "const_100": 100.0,
    "const_1000": 1000.0,
    "const_10000": 10000.0,
    "const_100000": 100000.0,
    "const_1000000": 1000000.0,
    "const_10000000": 10000000.0,
    "const_1000000000": 1000000000.0,
    "const_m1": -1.0,
}
STEP_RE = re.compile(r"^\s*([A-Za-z_]+)\((.*),(.*)\)\s*$")
PROGRAM_STEP_RE = re.compile(
    r"(?:add|subtract|multiply|divide|exp|greater)\([^()]*,[^()]*\)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?")


@dataclass(frozen=True)
class ProgramCheck:
    is_valid: bool
    error: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    value: float | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated FinQA reasoning programs.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--use-gold-programs", action="store_true", help="Create an oracle smoke-test from FinQA gold programs instead of Qwen outputs.")
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--arithmetic-only", action="store_true", help="Evaluate only examples whose gold program uses arithmetic operations, matching the Qwen arithmetic-program setup.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_program(program: str) -> str:
    program = normalize_text(program).lower()
    program = program.replace(" ", "")
    program = program.replace("const_m1", "const_m1")
    return program.rstrip(".")


def split_program_steps(program: str) -> list[str]:
    program = normalize_text(program)
    matches = PROGRAM_STEP_RE.findall(program)
    if matches:
        return [match.strip() for match in matches]
    return [step.strip() for step in program.split(",") if step.strip()]


def parse_step(step: str) -> tuple[str, str, str] | None:
    match = STEP_RE.match(step)
    if not match:
        return None
    op, arg1, arg2 = match.groups()
    return op.strip().lower(), arg1.strip(), arg2.strip()


def clean_number(value: str) -> str:
    return normalize_text(value).replace(",", "").replace(" ", "")


def parse_number(value: str) -> float | None:
    value = clean_number(value).lower()
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    try:
        number = float(value)
        return number / 100.0 if is_percent else number
    except ValueError:
        return None


def extract_program(raw_text: str) -> str:
    text = normalize_text(raw_text)
    text = re.sub(r"```(?:text|python)?", "", text, flags=re.IGNORECASE).replace("```", "")
    if "Program:" in text:
        text = text.split("Program:", 1)[-1].strip()
    matches = PROGRAM_STEP_RE.findall(text)
    if matches:
        return ", ".join(match.strip() for match in matches)
    return text.rstrip(".")


def validate_program(program: str) -> ProgramCheck:
    steps = split_program_steps(program)
    if not steps:
        return ProgramCheck(False, "empty program")
    for idx, step in enumerate(steps):
        parsed = parse_step(step)
        if parsed is None:
            return ProgramCheck(False, f"step is not op(arg1,arg2): {step}")
        op, arg1, arg2 = parsed
        if op not in ALLOWED_OPS:
            return ProgramCheck(False, f"operation is not allowed: {op}")
        for arg in [arg1, arg2]:
            if "(" in arg or ")" in arg:
                return ProgramCheck(False, f"nested calls are not allowed: {step}")
            if re.fullmatch(r"#\d+", arg):
                ref = int(arg[1:])
                if ref >= idx:
                    return ProgramCheck(False, f"{arg} does not reference an earlier step")
                continue
            if arg.lower() in CONST_VALUES:
                continue
            if parse_number(arg) is not None:
                continue
            return ProgramCheck(False, f"argument is not numeric/constant/prior step: {arg}")
    return ProgramCheck(True)


def resolve_arg(arg: str, results: list[float]) -> float:
    arg = arg.strip().lower()
    if re.fullmatch(r"#\d+", arg):
        return results[int(arg[1:])]
    if arg in CONST_VALUES:
        return CONST_VALUES[arg]
    value = parse_number(arg)
    if value is None:
        raise ValueError(f"cannot resolve argument: {arg}")
    return value


def execute_program(program: str) -> ExecutionResult:
    check = validate_program(program)
    if not check.is_valid:
        return ExecutionResult(False, error=check.error)
    results: list[float] = []
    try:
        for step in split_program_steps(program):
            parsed = parse_step(step)
            if parsed is None:
                raise ValueError(f"bad step: {step}")
            op, arg1, arg2 = parsed
            x = resolve_arg(arg1, results)
            y = resolve_arg(arg2, results)
            if op == "add":
                value = x + y
            elif op == "subtract":
                value = x - y
            elif op == "multiply":
                value = x * y
            elif op == "divide":
                if abs(y) < 1e-12:
                    raise ZeroDivisionError("divide by zero")
                value = x / y
            elif op == "exp":
                value = x ** y
            elif op == "greater":
                value = 1.0 if x > y else 0.0
            else:
                raise ValueError(f"unsupported op: {op}")
            results.append(value)
    except Exception as exc:  # noqa: BLE001 - we want the error in the CSV.
        return ExecutionResult(False, error=str(exc))
    return ExecutionResult(True, value=results[-1] if results else None)


def numeric_answer(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"yes", "true"}:
        return 1.0
    if lowered in {"no", "false"}:
        return 0.0
    parsed = parse_number(text)
    if parsed is not None:
        return parsed
    matches = NUMBER_RE.findall(text)
    if matches:
        return parse_number(matches[0])
    return None


def answers_close(predicted: float | None, gold: float | None, tolerance: float) -> bool:
    if predicted is None or gold is None:
        return False
    scale = max(1.0, abs(gold))
    return abs(predicted - gold) <= tolerance * scale


def first_operation(program: str) -> str:
    for step in split_program_steps(program):
        parsed = parse_step(step)
        if parsed:
            return parsed[0]
    return "unknown"


def question_category(question: str, gold_program: str) -> str:
    q = question.lower()
    op = first_operation(gold_program)
    if any(word in q for word in ["ratio", "percentage", "percent", "margin", "per "]):
        return "division/percentage"
    if any(word in q for word in ["increase", "decrease", "change", "difference", "more", "less"]):
        return "difference/change"
    if any(word in q for word in ["total", "combined", "sum", "aggregate"]):
        return "aggregation"
    if op == "divide":
        return "division/percentage"
    if op == "subtract":
        return "difference/change"
    if op == "add":
        return "aggregation"
    return "other"



def canonical_program_from_steps(steps: Any, fallback: str) -> str:
    if not isinstance(steps, list) or not steps:
        return normalize_text(fallback)
    out = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            return normalize_text(fallback)
        raw_op = normalize_text(step.get("op")).lower()
        op_match = re.match(r"[a-z_]+", raw_op)
        if not op_match:
            return normalize_text(fallback)
        op = op_match.group(0)
        op = {
            "minus": "subtract",
            "sum": "add",
            "average": "table_average",
            "max": "table_max",
            "min": "table_min",
            "compare_larger": "greater",
        }.get(op, op)
        arg1 = normalize_text(step.get("arg1"))
        arg2 = normalize_text(step.get("arg2"))
        if not arg1 or not arg2:
            return normalize_text(fallback)
        out.append(f"{op}({arg1}, {arg2})")
    return ", ".join(out)


def load_dataset(path: Path, max_examples: int | None) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if max_examples is not None:
        data = data[:max_examples]
    rows = {}
    for ex in data:
        qa = ex.get("qa", {})
        example_id = normalize_text(ex.get("id") or ex.get("filename"))
        fallback_program = normalize_text(qa.get("program_re") or qa.get("program"))
        rows[example_id] = {
            "example_id": example_id,
            "question": normalize_text(qa.get("question")),
            "gold_program": canonical_program_from_steps(qa.get("steps"), fallback_program),
            "gold_answer": normalize_text(qa.get("exe_ans") if qa.get("exe_ans") is not None else qa.get("answer")),
            "answer_text": normalize_text(qa.get("answer")),
        }
    return rows


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_arithmetic_program(program: str) -> bool:
    return "table_" not in normalize_program(program) and validate_program(program).is_valid


def make_gold_predictions(gold_rows: dict[str, dict], arithmetic_only: bool = False) -> list[dict]:
    rows = []
    for ex in gold_rows.values():
        if arithmetic_only and not is_arithmetic_program(ex["gold_program"]):
            continue
        rows.append(
            {
                "query_example_id": ex["example_id"],
                "query": ex["question"],
                "generated_program": ex["gold_program"],
                "retrieval_source": "gold_program_smoke_test",
            }
        )
    return rows


def prediction_example_id(row: dict) -> str:
    return normalize_text(row.get("query_example_id") or row.get("example_id") or row.get("id"))


def prediction_program(row: dict) -> str:
    return extract_program(row.get("generated_program") or row.get("program") or row.get("prediction") or row.get("raw_generation") or "")


def prediction_source(row: dict) -> str:
    return normalize_text(row.get("retrieval_source") or row.get("mode") or row.get("candidate_scope") or "qwen_generated")


def evaluate_rows(predictions: list[dict], gold_rows: dict[str, dict], tolerance: float) -> list[dict]:
    out = []
    for row in predictions:
        example_id = prediction_example_id(row)
        if example_id not in gold_rows:
            continue
        gold = gold_rows[example_id]
        generated = prediction_program(row)
        valid = validate_program(generated)
        execution = execute_program(generated)
        predicted_answer = execution.value if execution.ok else None
        gold_answer = numeric_answer(gold["gold_answer"])
        exact = normalize_program(generated) == normalize_program(gold["gold_program"])
        answer_ok = answers_close(predicted_answer, gold_answer, tolerance)
        out.append(
            {
                "example_id": example_id,
                "retrieval_source": prediction_source(row),
                "question": gold["question"],
                "gold_program": gold["gold_program"],
                "generated_program": generated,
                "gold_answer": gold["gold_answer"],
                "predicted_answer": "" if predicted_answer is None else round(predicted_answer, 8),
                "gold_operation": first_operation(gold["gold_program"]),
                "generated_operation": first_operation(generated),
                "question_category": question_category(gold["question"], gold["gold_program"]),
                "is_valid_program": valid.is_valid,
                "validation_error": valid.error,
                "program_exact_match": exact,
                "execution_ok": execution.ok,
                "execution_error": execution.error,
                "answer_accuracy": answer_ok,
            }
        )
    return out


def summarize(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalize_text(row[group_key])].append(row)
    summary = []
    for group, items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                group_key: group,
                "n": n,
                "program_validity_rate": round(sum(bool(r["is_valid_program"]) for r in items) / n, 4),
                "program_exact_match_rate": round(sum(bool(r["program_exact_match"]) for r in items) / n, 4),
                "execution_rate": round(sum(bool(r["execution_ok"]) for r in items) / n, 4),
                "answer_accuracy": round(sum(bool(r["answer_accuracy"]) for r in items) / n, 4),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def plot_metric_bars(rows: list[dict], label_key: str, metric_keys: list[str], title: str, path: Path) -> None:
    width, height = 980, 560
    margin = 78
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]
    labels = [r[label_key] for r in rows]
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(metric_keys) + 1.4)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
    ]
    for i in range(6):
        y_val = i / 5
        y = height - margin - y_val * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.1f}</text>')
    for gi, row in enumerate(rows):
        x0 = margin + gi * group_w + group_w * 0.12
        for mi, metric in enumerate(metric_keys):
            value = float(row[metric])
            bar_h = value * plot_h
            x = x0 + mi * bar_w
            y = height - margin - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5:.1f}" height="{bar_h:.1f}" fill="{colors[mi % len(colors)]}"/>')
            parts.append(f'<text x="{x + (bar_w-5)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="10" font-family="Arial">{value:.0%}</text>')
        parts.append(f'<text x="{margin + gi * group_w + group_w/2:.1f}" y="{height-margin+28}" text-anchor="middle" font-size="12" font-family="Arial">{svg_escape(row[label_key])}</text>')
    legend_x = width - margin - 250
    for mi, metric in enumerate(metric_keys):
        y = 78 + mi * 24
        parts.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" fill="{colors[mi % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x+22}" y="{y}" font-size="13" font-family="Arial">{svg_escape(metric.replace("_", " "))}</text>')
    parts.append('<text x="24" y="280" transform="rotate(-90 24 280)" text-anchor="middle" font-size="13" font-family="Arial">Rate</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(rows: list[dict], source_summary: list[dict], op_summary: list[dict], category_summary: list[dict], output_dir: Path) -> None:
    total = len(rows)
    valid = sum(bool(r["is_valid_program"]) for r in rows)
    exact = sum(bool(r["program_exact_match"]) for r in rows)
    exec_ok = sum(bool(r["execution_ok"]) for r in rows)
    answer = sum(bool(r["answer_accuracy"]) for r in rows)
    op_counts = Counter(r["gold_operation"] for r in rows)
    lines = [
        "# FinQA Program Generation Evaluation",
        "",
        f"Evaluated programs: {total}",
        "",
        "## Overall Metrics",
        "Arithmetic-only subset: yes" if any(r.get("retrieval_source") == "gold_program_smoke_test" for r in rows) else "Arithmetic-only subset: see command used",
        f"- Program validity: {valid / total:.1%}" if total else "- Program validity: n/a",
        f"- Program exact match: {exact / total:.1%}" if total else "- Program exact match: n/a",
        f"- Execution rate: {exec_ok / total:.1%}" if total else "- Execution rate: n/a",
        f"- Answer accuracy: {answer / total:.1%}" if total else "- Answer accuracy: n/a",
        "",
        "## Operation Mix",
    ]
    for op, count in op_counts.most_common():
        lines.append(f"- {op}: {count}")
    lines.extend(["", "## Source Summary"])
    for row in source_summary:
        lines.append(
            f"- {row['retrieval_source']}: validity {float(row['program_validity_rate']):.1%}, "
            f"exact {float(row['program_exact_match_rate']):.1%}, answer {float(row['answer_accuracy']):.1%} (n={row['n']})"
        )
    lines.extend(["", "## Files"])
    lines.extend(
        [
            "- outputs/generation/program_evaluation_details.csv",
            "- outputs/generation/program_evaluation_by_source.csv",
            "- outputs/generation/program_evaluation_by_operation.csv",
            "- outputs/generation/program_evaluation_by_category.csv",
            "- plots/generation/program_metrics_by_source.svg",
            "- plots/generation/answer_accuracy_by_operation.svg",
            "- plots/generation/answer_accuracy_by_category.svg",
        ]
    )
    (output_dir / "program_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = load_dataset(args.dataset, args.max_examples)
    if args.use_gold_programs:
        predictions = make_gold_predictions(gold_rows, arithmetic_only=args.arithmetic_only)
        prediction_source = "gold_program_smoke_test"
    else:
        if not args.predictions.exists():
            raise FileNotFoundError(
                f"Missing predictions file: {args.predictions}. Run Qwen generation first, "
                "or use --use-gold-programs for a runnable smoke test."
            )
        predictions = read_predictions(args.predictions)
        prediction_source = str(args.predictions)

    if args.arithmetic_only:
        arithmetic_ids = {ex_id for ex_id, ex in gold_rows.items() if is_arithmetic_program(ex["gold_program"])}
        predictions = [row for row in predictions if prediction_example_id(row) in arithmetic_ids]

    details = evaluate_rows(predictions, gold_rows, args.tolerance)
    if not details:
        raise RuntimeError("No predictions matched dataset example ids.")

    source_summary = summarize(details, "retrieval_source")
    op_summary = summarize(details, "gold_operation")
    category_summary = summarize(details, "question_category")

    write_csv(args.output_dir / "program_evaluation_details.csv", details)
    write_csv(args.output_dir / "program_evaluation_by_source.csv", source_summary)
    write_csv(args.output_dir / "program_evaluation_by_operation.csv", op_summary)
    write_csv(args.output_dir / "program_evaluation_by_category.csv", category_summary)

    plot_metric_bars(
        source_summary,
        "retrieval_source",
        ["program_validity_rate", "program_exact_match_rate", "answer_accuracy"],
        "Program Generation Metrics by Evidence Source",
        args.plot_dir / "program_metrics_by_source.svg",
    )
    plot_metric_bars(
        op_summary,
        "gold_operation",
        ["answer_accuracy", "program_exact_match_rate"],
        "Numerical Accuracy by Operation",
        args.plot_dir / "answer_accuracy_by_operation.svg",
    )
    plot_metric_bars(
        category_summary,
        "question_category",
        ["answer_accuracy", "program_exact_match_rate"],
        "Numerical Accuracy by Question Type",
        args.plot_dir / "answer_accuracy_by_category.svg",
    )
    write_summary(details, source_summary, op_summary, category_summary, args.output_dir)

    print(f"Input predictions: {prediction_source}")
    print(f"Evaluated programs: {len(details)}")
    print(f"Program validity: {sum(bool(r['is_valid_program']) for r in details) / len(details):.1%}")
    print(f"Program exact match: {sum(bool(r['program_exact_match']) for r in details) / len(details):.1%}")
    print(f"Execution rate: {sum(bool(r['execution_ok']) for r in details) / len(details):.1%}")
    print(f"Answer accuracy: {sum(bool(r['answer_accuracy']) for r in details) / len(details):.1%}")
    print(f"Saved details to {(args.output_dir / 'program_evaluation_details.csv').resolve()}")
    print(f"Saved plots to {args.plot_dir.resolve()}")


if __name__ == "__main__":
    main()
