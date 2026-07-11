"""Shared infrastructure for in-house baseline reproductions.

Goal: every baseline trains *exactly the same student architecture* on *the
same corpus and token budget* as the FSD run. Only the loss function differs.
This makes the headline comparison apples-to-apples.

Use case::

    from scripts.repro_baselines._common import (
        BaselineArgs, load_teacher_and_tokenizer, load_corpus, build_matched_student,
        train_loop, save_run,
    )

    args = parse_baseline_args()  # base BaselineArgs + your loss-specific knobs
    teacher, tok = load_teacher_and_tokenizer(args)
    student = build_matched_student(args, teacher)  # reads FSD student arch from disk
    train_loader = load_corpus(args, tok)

    def loss_fn(s_logits, t_logits, batch):
        return your_loss(s_logits, t_logits)  # the one thing each baseline customises

    train_loop(student, teacher, train_loader, args, loss_fn)
    save_run(student, args)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class BaselineArgs:
    teacher: str
    student_config: str  # path to FSD's saved student config (locks architecture)
    corpus: str
    tokens_per_rung: int
    seq_len: int
    per_gpu_batch: int
    grad_accum: int
    lr: float
    seed: int
    output_dir: str
    dry_run: bool


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--student-config", type=str, required=True,
                   help="Path to a saved FSD student-config JSON (so this baseline "
                        "uses the exact same architecture).")
    p.add_argument("--corpus", type=str, default="slimpajama")
    p.add_argument("--tokens-per-rung", type=int, default=10_000_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--per-gpu-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--dry-run", action="store_true")


def args_from_namespace(ns: argparse.Namespace) -> BaselineArgs:
    return BaselineArgs(
        teacher=ns.teacher,
        student_config=ns.student_config,
        corpus=ns.corpus,
        tokens_per_rung=ns.tokens_per_rung,
        seq_len=ns.seq_len,
        per_gpu_batch=ns.per_gpu_batch,
        grad_accum=ns.grad_accum,
        lr=ns.lr,
        seed=ns.seed,
        output_dir=ns.output_dir,
        dry_run=ns.dry_run,
    )


def load_teacher_and_tokenizer(args: BaselineArgs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[baseline] Loading teacher {args.teacher}...")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16
    )
    teacher.eval()
    return teacher, tok


def build_matched_student(args: BaselineArgs, teacher) -> nn.Module:
    """Construct a student with the SAME architecture as the FSD run.

    Reads the saved StudentConfig (hidden, intermediate, heads, kv-heads, layers)
    and instantiates a fresh randomly-initialised model of that shape. No
    absorbed init — that's FSD's contribution. Baselines start from random weights.
    """
    with open(args.student_config) as f:
        cfg = json.load(f)
    teacher_cls = type(teacher).__name__
    if "Llama" in teacher_cls or "Qwen" in teacher_cls or "Mistral" in teacher_cls:
        from transformers import LlamaConfig, LlamaForCausalLM
        s_cfg = LlamaConfig(
            vocab_size=int(teacher.config.vocab_size),
            max_position_embeddings=int(getattr(teacher.config, "max_position_embeddings", 2048)),
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            rms_norm_eps=float(getattr(teacher.config, "rms_norm_eps", 1e-6)),
            rope_theta=float(getattr(teacher.config, "rope_theta", 10000.0)),
            hidden_act=getattr(teacher.config, "hidden_act", "silu"),
        )
        return LlamaForCausalLM(s_cfg)
    if "GPT2" in teacher_cls:
        from transformers import GPT2Config, GPT2LMHeadModel
        s_cfg = GPT2Config(
            vocab_size=int(teacher.config.vocab_size),
            n_positions=int(getattr(teacher.config, "n_positions", 1024)),
            n_embd=cfg["hidden_size"],
            n_layer=cfg["num_hidden_layers"],
            n_head=cfg["num_attention_heads"],
            n_inner=cfg["intermediate_size"],
        )
        return GPT2LMHeadModel(s_cfg)
    raise NotImplementedError(teacher_cls)


def load_corpus(args: BaselineArgs, tokenizer):
    from datasets import load_dataset
    if args.corpus == "slimpajama":
        ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    elif args.corpus == "dclm":
        ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
    elif args.corpus == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    else:
        raise ValueError(args.corpus)
    if args.dry_run:
        ds = ds.take(64)

    def collate(batch):
        texts = [b["text"] for b in batch if b.get("text")]
        if not texts:
            return None
        return tokenizer(texts, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=args.seq_len)

    return DataLoader(ds, batch_size=args.per_gpu_batch, collate_fn=collate)


def train_loop(student, teacher, train_loader, args: BaselineArgs, loss_fn):
    """Generic training loop: each baseline supplies its loss function.

    ``loss_fn(s_logits, t_logits, batch) -> Tensor``: the per-step loss.

    Optimiser is plain AdamW, same hyperparameters across all baselines.
    """
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    student.to("cuda" if torch.cuda.is_available() else "cpu").train()
    teacher.eval()

    step = 0
    total_steps = (
        100 if args.dry_run else
        max(1, args.tokens_per_rung // (args.per_gpu_batch * args.seq_len))
    )

    for batch in train_loader:
        if batch is None:
            continue
        if step >= total_steps:
            break
        device = next(student.parameters()).device
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits
        s_logits = student(input_ids=ids).logits

        loss = loss_fn(s_logits, t_logits, batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[baseline] step={step}  loss={loss.item():.4f}")
        step += 1

    return student


def save_run(student, args: BaselineArgs) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save(student.state_dict(), out / "student.pt")
    print(f"[baseline] saved to {out}")
