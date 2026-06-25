from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "finqa_topk_results_same_example.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "finqa_qwen3_programs.csv"
DEFAULT_CONSTANTS_FILE = PROJECT_ROOT / "FinQA" / "code" / "retriever" / "constant_list.txt"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"

ALLOWED_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
STEP_RE = re.compile(r"^\s*([A-Za-z_]+)\((.*),(.*)\)\s*$")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?")
PROGRAM_STEP_RE = re.compile(
    r"(?:add|subtract|multiply|divide|exp|greater)\([^()]*,[^()]*\)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen3-8B to generate FinQA reasoning programs from a question "
            "and the top three retrieved evidence rows."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--constants-file", type=Path, default=DEFAULT_CONSTANTS_FILE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--evidence-rows", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to transformers.from_pretrained. Use 'cpu' to force CPU.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_offload",
        help="Folder used when transformers offloads model weights to disk.",
    )
    parser.add_argument(
        "--allow-meta-weights",
        action="store_true",
        help="Skip the post-load check for meta tensors. Usually not recommended.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model/tokenizer.",
    )
    return parser.parse_args()


def load_constants(path: Path) -> set[str]:
    constants = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#") or token == "NONE":
            continue
        if token.startswith("#"):
            continue
        constants.add(token)
        constants.add(token.lower())
    return constants


def normalize_number(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", "")


def evidence_numbers(evidence_texts: list[str]) -> set[str]:
    nums = set()
    for text in evidence_texts:
        for match in NUMBER_RE.findall(str(text)):
            normalized = normalize_number(match)
            nums.add(normalized)
            if normalized.endswith("%"):
                nums.add(normalized[:-1])
    return nums


def build_prompt(question: str, evidence_texts: list[str], constants: set[str]) -> str:
    constants_for_prompt = sorted(
        token for token in constants if token.upper().startswith("CONST_") and token == token.upper()
    )
    evidence_block = "\n".join(
        f"{idx}. {text}" for idx, text in enumerate(evidence_texts, start=1)
    )
    constants_block = ", ".join(constants_for_prompt)
    return f"""You will receive a financial question and associated evidence.
    Your goal is to generate an arithmetic reasoning program that would execute
    to give the correct final numeric/logical answer to the question.
   
    Here are 8 example problems. Remember, your goal is to generate the reasoning program.

EXAMPLE 1:
    Question: What was the percentage increase in company income from 2019 to 2020?
    Evidence 1: Company income in year 2019 was 300 million.
    Evidence 2: Company income in year 2020 was 350 million.
    Evidence 3: Toothbrushes cost 5 dollars!
    Generated reasoning program: subtract(350,300), divide(#0,300), multiply(#1,CONST_100)
                                                           
EXAMPLE 2:
    Question: what is the net chance in unrecognized tax benefits from 2011 to 2012 , ( in millions ) ?
    Evidence 1: the utilization of these net operating losses is subject to certain annual limitations as required under internal revenue code section 382 and similar state income tax provisions .
    Evidence 2: the company 2019s gross unrecognized tax benefits totaled $ 52.4 million and $ 32.1 million as of september 28 , 2012 and september 30 , 2011 , respectively .
    Generated reasoning program: subtract(52.4, 32.1)
   
EXAMPLE 3:
    Question: what is the total in millions of current assets acquired?
    Evidence 1: cash the accounts receivable of $ 116 is 278 ;
    Evidence 2: cash the inventory of $ 116 is 124 ;
    Evidence 3: cash the other current assets of $ 116 is 41 ;
    Generated reasoning program: add(116, 278), add(#0, 124), add(#1, 41)
                                                   
EXAMPLE 4:
    Question: considering the years 2012 and 2013 , what is the increase observed in the balance at the end of the year?
    Evidence 1: unrecognized tax benefits the balance at end of year of 2013 is $ 124.3 ; the balance at end of year of 2012 is $ 110.8 ; the balance at end of year of 2011 is $ 126.4 ;
    Generated reasoning program: divide(124.3, 110.8), subtract(#0, CONST_1)
                                                               
EXAMPLE 5:
    Question: during the 2012 year , did the equity awards in which the prescribed performance milestones were achieved exceed the equity award compensation expense for equity granted during the year?
    Evidence 1: the granted of number of shares ( in thousands ) is 607 ; the granted of weighted average grant date fair value ( per share ) is 18.13 ;
    Evidence 2: during the year ended march 31 , 2012 , the company has recorded $ 3.3 million in stock-based compensation expense for equity awards in which the prescribed performance milestones have been achieved or are probable of being achieved .
    Generated reasoning program: multiply(607, 18.13), multiply(#0, CONST_1000), multiply(3.3, CONST_1000000), greater(#1, #2)

EXAMPLE 6:
    Question: what percentage decrease occurred from 2011-2012 for deferred acquisition payments?                                                                
    Evidence 1: the deferred acquisition payments of 2010 is $ 20.5 ; the deferred acquisition payments of 2011 is $ 34.8 ; the deferred acquisition payments of 2012 is $ 1.2 ; the deferred acquisition payments of 2013 is $ 1.1 ; the deferred acquisition payments of 2014 is $ 2.1 ; the deferred acquisition payments of thereafter is $ 0.3 ; the deferred acquisition payments of total is $ 60.0 ;
    Evidence 2: all payments are contingent upon achieving projected operating performance targets and satisfying other conditions specified in the related agreements and are subject to revisions as the earn-out periods progress. .
    Generated reasoning program: subtract(34.8, 1.2), divide(#0, 34.8), multiply(#1, CONST_100)
                                                             
EXAMPLE 7:
    Question: what is the percentage increase in obligation for the mrrp from 2011 to 2012?
    Evidence 1: at december 31 , 2012 and 2011 , the obligation for the mrrp totaled $ 22.7 million and $ 21.6 million , respectively .
    Generated reasoning program: subtract(22.7, 21.6), divide(#0, 21.6)
                                                       
EXAMPLE 8:
    Question: what percent of total contractual obligations is due to long-term debt ( including interest ) ?
    Evidence 1: ( dollars in millions ) the long-term debt ( including interest ) of amounts due by period total is $ 5342 ; the long-term debt ( including interest ) of amounts due by period less than 1 year is 428 ; the long-term debt ( including interest ) of amounts due by period 1 - 3years is 1434 ; the long-term debt ( including interest ) of amounts due by period 3 - 5years is 966 ; the long-term debt ( including interest ) of amounts due by period more than5 years is 2514 ;
    Evidence 2: ( dollars in millions ) the total of amounts due by period total is $ 6624 ; the total of amounts due by period less than 1 year is 1254 ; the total of amounts due by period 1 - 3years is 1711 ; the total of amounts due by period 3 - 5years is 1060 ; the total of amounts due by period more than5 years is 2599 ;
    Generated reasoning program: divide(5342, 6624)


   
END EXAMPLES.
                                                       
Your final generated reasoning program should be of the form "op1(arg1,arg2), op2(arg3,arg4), ...", where:
    the "op"s are chosen from the "Allowed operations" outlined below, and
    the "arg"s are all numbers that are STRICTLY chosen from either the "Evidence" OR the "Allowed constants" outlined below OR a previous operation result.
    If you need to reference the numeric result of a previous operation in later operations, you can denote the numeric result of op1 as "#0" and numeric result of op2 as "#1" and so forth...
    Your answer should ONLY be op1(arg1,arg2), op2(arg3,arg4), ...

    Here are the financial question to be answered, the evidence to use to answer it, the allowed math operations, and the allowed constants.

Question:
{question}

Evidence:
{evidence_block}

Allowed operations:
add, subtract, multiply, divide, exp, greater

Allowed constants:
{constants_block}

Use CONST_M1 to denote negative 1 (-1) if necessary.                                                      
"""


def build_supportability_prompt(question: str, evidence_texts: list[str]) -> str:
    evidence_block = "\n".join(
        f"{idx}. {text}" for idx, text in enumerate(evidence_texts, start=1)
    )
    return f"""Question:
{question}

Retrieved Evidence:

{evidence_block}

Instructions:
Determine how well the retrieved evidence supports answering the question.

Return exactly ONE label:

STRONGLY_SUPPORTED
WEAKLY_SUPPORTED
NOT_SUPPORTED

Do not provide explanations.
Do not generate any additional text.
Return only the label."""


def extract_supportability_label(raw_text: str) -> str:
    text = raw_text or ""
    text = re.sub(r"```(?:text|python)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text).strip().upper()

    for label in ("STRONGLY_SUPPORTED", "WEAKLY_SUPPORTED", "NOT_SUPPORTED"):
        if re.search(rf"\b{label}\b", text):
            return label
    return "INVALID"


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise financial reasoning program generator. "
                "Return only the requested program."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_input_device(model: Any) -> Any:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for module_name in ["model.embed_tokens", "embed_tokens"]:
            device = hf_device_map.get(module_name)
            if device is not None and str(device) != "meta":
                return device
        for device in hf_device_map.values():
            if str(device) not in {"meta", "disk"}:
                return device

    for param in model.parameters():
        if str(param.device) != "meta":
            return param.device

    return "cpu"


def find_meta_parameters(model: Any, max_names: int = 10) -> list[str]:
    names = []
    for name, param in model.named_parameters():
        if str(param.device) == "meta":
            names.append(name)
            if len(names) >= max_names:
                break
    return names


def generate_program(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(get_input_device(model))
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    new_ids = generated_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def extract_program(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"```(?:text|python)?", "", text, flags=re.IGNORECASE).replace("```", "")
    text = text.replace("\n", " ").strip()
    if "Program:" in text:
        text = text.split("Program:", 1)[-1].strip()

    matches = PROGRAM_STEP_RE.findall(text)
    if matches:
        return ", ".join(match.strip() for match in matches)
    return text.strip().rstrip(".")


def split_program_steps(program: str) -> list[str]:
    matches = PROGRAM_STEP_RE.findall(program)
    if matches:
        return [match.strip() for match in matches]
    return [step.strip() for step in program.split(",") if step.strip()]


def validate_arg(
    arg: str,
    step_index: int,
    numbers: set[str],
    constants: set[str],
) -> ValidationResult:
    arg = arg.strip()
    if re.fullmatch(r"#\d+", arg):
        ref_index = int(arg[1:])
        if ref_index >= step_index:
            return ValidationResult(False, f"{arg} does not reference an earlier step")
        return ValidationResult(True, "")

    if arg in constants or arg.upper() in constants:
        return ValidationResult(True, "")

    normalized = normalize_number(arg)
    if normalized in numbers:
        return ValidationResult(True, "")

    return ValidationResult(False, f"argument is not from evidence/constants/prior steps: {arg}")


def validate_program(
    program: str,
    evidence_texts: list[str],
    constants: set[str],
) -> ValidationResult:
    steps = split_program_steps(program)
    if not steps:
        return ValidationResult(False, "empty program")

    nums = evidence_numbers(evidence_texts)
    for step_index, step in enumerate(steps):
        match = STEP_RE.match(step)
        if not match:
            return ValidationResult(False, f"step is not op(arg1,arg2): {step}")
        op, arg1, arg2 = match.groups()
        op = op.strip().lower()
        if op not in ALLOWED_OPS:
            return ValidationResult(False, f"operation is not allowed: {op}")
        if "(" in arg1 or ")" in arg1 or "(" in arg2 or ")" in arg2:
            return ValidationResult(False, f"nested calls are not allowed: {step}")

        arg1_valid = validate_arg(arg1, step_index, nums, constants)
        if not arg1_valid.is_valid:
            return arg1_valid
        arg2_valid = validate_arg(arg2, step_index, nums, constants)
        if not arg2_valid.is_valid:
            return arg2_valid

    return ValidationResult(True, "")


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": "auto",
    }
    if args.device_map == "cpu":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["offload_folder"] = str(args.offload_folder)
        model_kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.device_map == "cpu":
        model = model.to("cpu")
    meta_names = find_meta_parameters(model)
    if meta_names and not args.allow_meta_weights:
        device_map = getattr(model, "hf_device_map", None)
        raise RuntimeError(
            "The model still has parameters on the meta device after loading. "
            "This usually means Qwen3-8B did not fit in the available RAM/VRAM "
            "with the current device_map. First meta parameters: "
            f"{meta_names}. Device map: {device_map}. Try one of: "
            "1) run with --device-map cpu, which is slow but avoids auto sharding; "
            "2) use a smaller/quantized model; "
            "3) install/configure bitsandbytes and add quantized loading support."
        )
    model.eval()
    return model, tokenizer


def iter_question_groups(df: pd.DataFrame, evidence_rows: int) -> list[tuple[str, pd.DataFrame]]:
    required_cols = {"query_example_id", "query", "retrieved_text", "rank"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    groups = []
    for example_id, group in df.sort_values(["query_example_id", "rank"]).groupby("query_example_id"):
        groups.append((example_id, group.head(evidence_rows).copy()))
    return groups


def main() -> None:
    args = parse_args()
    constants = load_constants(args.constants_file)

    df = pd.read_csv(args.input_csv)
    groups = iter_question_groups(df, args.evidence_rows)
    if args.max_examples is not None:
        groups = groups[: args.max_examples]

    print(f"Loaded {len(groups)} question groups from {args.input_csv}")
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)

    rows_out = []
    for group_index, (example_id, group) in enumerate(groups, start=1):
        question = str(group.iloc[0]["query"])
        evidence_texts = [str(text) for text in group["retrieved_text"].tolist()]
        prompt = build_prompt(question, evidence_texts, constants)
        raw_generation = generate_program(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        program = extract_program(raw_generation)
        validation = validate_program(program, evidence_texts, constants)

        support_prompt = build_supportability_prompt(question, evidence_texts)
        supportability_raw_generation = generate_program(
            model,
            tokenizer,
            support_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        supportability_label = extract_supportability_label(supportability_raw_generation)

        rows_out.append(
            {
                "query_example_id": example_id,
                "query": question,
                "evidence_1": evidence_texts[0] if len(evidence_texts) > 0 else "",
                "evidence_2": evidence_texts[1] if len(evidence_texts) > 1 else "",
                "evidence_3": evidence_texts[2] if len(evidence_texts) > 2 else "",
                "generated_program": program,
                "raw_generation": raw_generation,
                "is_valid_program": validation.is_valid,
                "validation_error": validation.error,
                "supportability_label": supportability_label,
                "supportability_raw_generation": supportability_raw_generation,
            }
        )
        print(
            f"[{group_index}/{len(groups)}] {example_id}: "
            f"{'valid' if validation.is_valid else 'invalid'}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(args.output_csv, index=False)

    support_counts = out_df["supportability_label"].value_counts(dropna=False)
    print("Supportability label counts:")
    for label in ["STRONGLY_SUPPORTED", "WEAKLY_SUPPORTED", "NOT_SUPPORTED", "INVALID"]:
        if label in support_counts:
            print(f"- {label}: {support_counts[label]}")
    print(f"Saved generated programs to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
