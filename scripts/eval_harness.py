"""Wrapper around lm-evaluation-harness for FSD's headline numbers.

Sprint 7's headline frontier is **zero-shot harness average vs. distillation
token budget**. This script evaluates a saved student checkpoint on the
canonical task suite and emits a JSON summary.

Tasks (the standard "small-LM zero-shot" set used by Sheared-Llama et al.):
  - HellaSwag (commonsense)
  - ARC-easy / ARC-challenge (science Q&A)
  - PIQA (physical reasoning)
  - WinoGrande (coreference)
  - MMLU (5-shot, knowledge)
  - LAMBADA (long-distance language modelling)
  - OpenBookQA (open-book Q&A)
  - BoolQ (yes/no reading comprehension)

PPL is reported separately via WikiText-103 + C4-validation (smaller, paired
with the distillation corpus).

Requires::

    pip install lm-eval

Usage::

    python scripts/eval_harness.py \
        --student-dir runs/fsd_llama32_3b_to_1b_10B_seed0 \
        --tokenizer meta-llama/Llama-3.2-3B \
        --output runs/fsd_llama32_3b_to_1b_10B_seed0/eval.json

Or as a sweep across multiple runs::

    python scripts/eval_harness.py \
        --runs 'runs/*_llama32_*_10B_seed*' \
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu \
        --output eval/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

DEFAULT_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "mmlu",
    "lambada_openai",
    "openbookqa",
    "boolq",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student-dir", type=str, default=None,
                   help="Single student-checkpoint directory (config.json + student.pt).")
    p.add_argument("--runs", type=str, default=None,
                   help="Glob pattern matching multiple run directories.")
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    p.add_argument("--num-fewshot", type=int, default=0,
                   help="Default n-shot. MMLU is 5-shot; per-task overrides applied via lm-eval.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, evaluate only this many examples per task (for smoke tests).")
    return p.parse_args()


def evaluate_one(student_dir: str, args, tasks: list[str]) -> dict:
    """Load a student checkpoint and evaluate on lm-eval tasks."""
    try:
        import lm_eval
        import torch
        from lm_eval.models.huggingface import HFLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "eval_harness requires `pip install lm-eval transformers`"
        ) from e

    cfg_path = Path(student_dir) / "config.json"
    weights_path = Path(student_dir) / "student.pt"

    print(f"[eval] Loading student from {student_dir}...")
    with open(cfg_path) as f:
        student_cfg = json.load(f)

    teacher_name = student_cfg.get("teacher", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(teacher_name)
    # Reconstruct architecture from saved config — we use the teacher's class
    # with overridden config to match the student shape.
    base = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.float32)
    # Replace weights with student state_dict. (For non-trivial arch differences
    # — e.g. compressed hidden — load_state_dict expects matched shapes; that's
    # why we save the *student-shape* model, not the teacher.)
    state = torch.load(weights_path, map_location="cpu")
    base.load_state_dict(state, strict=False)
    base.to(args.device)

    lm = HFLM(pretrained=base, tokenizer=tok, batch_size=args.batch_size)
    print(f"[eval] Running tasks: {tasks}")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # Distill the harness output to a flat metric dict.
    summary = {}
    for task, metrics in results.get("results", {}).items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                summary[f"{task}/{k}"] = v
    summary["_run_dir"] = student_dir
    summary["_tasks"] = tasks
    return summary


def main() -> int:
    args = parse_args()
    tasks = args.tasks.split(",")

    if args.student_dir:
        runs = [args.student_dir]
    elif args.runs:
        runs = sorted(glob.glob(args.runs))
    else:
        print("error: pass --student-dir or --runs", file=sys.stderr)
        return 1

    all_results = []
    for run_dir in runs:
        try:
            summary = evaluate_one(run_dir, args, tasks)
        except Exception as e:
            print(f"[eval] {run_dir} FAILED: {e}", file=sys.stderr)
            continue
        all_results.append(summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[eval] Wrote {len(all_results)} results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
