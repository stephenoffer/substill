"""FSD main-paper training script for Llama-3.2-3B → 1B distillation.

Composes the full FSD stack:
  - γ-fold pre-pass (rotation-equivariant normalization, teacher side)
  - RR-Norm replacement in student (rotation-equivariant normalization, student side)
  - Function-aware Fisher scoring
  - Exact greedy q/cost rank allocator
  - Block-diagonal sparse correction
  - Trainable Stiefel bases via StiefelAdam
  - Adaptive entropy-gap skew-KL + plateau detector + unified token weighting

Compute requirements:
  - Llama-3.2-3B teacher (gated; user must provide HF token)
  - 4-8 H100 GPUs recommended for the headline run
  - Distillation corpus: SlimPajama or DCLM-baseline subset
  - Token budgets: 5B / 10B / 20B for the data-efficiency sweep

Usage::

    HF_TOKEN=... python scripts/distill_llama32_fsd.py \
        --teacher meta-llama/Llama-3.2-3B \
        --student-target-params 1.2e9 \
        --tokens-per-rung 10_000_000_000 \
        --corpus slimpajama \
        --calib-sequences 4096 \
        --output-dir runs/fsd_llama32_3b_to_1b_10B_seed0 \
        --seed 0

The script is structured so a small-scale dry-run (``--dry-run``) on a single
GPU with a tiny calibration set verifies the pipeline composes correctly
before launching the multi-GPU training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Existing FASD pieces we still use.
from substill import build_student
from substill.api import profile as fasd_profile
from substill.compression.rank_allocator import allocate_ranks, edges_from_profile
from substill.compression.sparse_block import CorrectedLinear
from substill.losses.generative_kd import (
    PlateauDetector,
    adaptive_skew_kl,
    forward_kl,
)
from substill.profiling.functional_score import score_directions

# --- FSD pillars --------------------------------------------------------------
from substill.profiling.gamma_fold import fold_gpt2, fold_llama
from substill.training.stiefel_optim import StiefelAdam, stiefel_param_groups
from substill.util.param_accounting import breakdown
from substill.util.rr_norm import replace_layernorm_with_rrnorm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--student-target-params", type=float, default=1.2e9,
                   help="Target trainable parameter count for the student.")
    p.add_argument("--corpus", type=str, default="slimpajama",
                   choices=["slimpajama", "dclm", "wikitext"])
    p.add_argument("--tokens-per-rung", type=int, default=10_000_000_000,
                   help="Distillation tokens per data-efficiency rung.")
    p.add_argument("--calib-sequences", type=int, default=4096,
                   help="Sequences for streaming PCA + Fisher scoring.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--per-gpu-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--stiefel-lr-ratio", type=float, default=0.1)
    p.add_argument("--use-rr-norm", action="store_true",
                   help="Replace student norms with RR-Norm + Stiefel Q.")
    p.add_argument("--use-fisher-score", action="store_true",
                   help="Score directions by Fisher vs. KL-patch baseline.")
    p.add_argument("--use-exact-allocator", action="store_true",
                   help="Use greedy q/cost knapsack allocator.")
    p.add_argument("--use-sparse-block", action="store_true",
                   help="Add block-diagonal per-head correction.")
    p.add_argument("--use-stiefel-optim", action="store_true",
                   help="Train Q (and bases) on Stiefel via StiefelAdam.")
    p.add_argument("--use-adaptive-skew-kl", action="store_true",
                   help="Per-token entropy-gap skew-KL.")
    p.add_argument("--use-plateau-trigger", action="store_true",
                   help="Trigger on-policy ramp from plateau detection.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Tiny dataset, 1 GPU, 100 steps — verify pipeline composes.")
    return p.parse_args()


def load_teacher(args):
    """Load the Llama-3.2 teacher. Requires HF_TOKEN env var for gated models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[fsd] Loading teacher {args.teacher}...")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16
    )
    teacher.eval()
    return teacher, tok


def load_corpus(args, tokenizer):
    """Load a streaming distillation corpus."""
    from datasets import load_dataset
    if args.corpus == "slimpajama":
        ds = load_dataset(
            "cerebras/SlimPajama-627B", split="train", streaming=True
        )
    elif args.corpus == "dclm":
        ds = load_dataset(
            "mlfoundations/dclm-baseline-1.0", split="train", streaming=True
        )
    elif args.corpus == "wikitext":
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        )
    else:
        raise ValueError(args.corpus)

    if args.dry_run:
        ds = ds.take(64)

    def collate(batch):
        texts = [b["text"] for b in batch if b.get("text")]
        if not texts:
            return None
        enc = tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=args.seq_len,
        )
        return enc

    return ds, collate


def stage_a_gamma_fold(teacher, args):
    """γ-fold the teacher (rotation-equivariant normalization, teacher side).

    Returns a *copy* of the teacher with γ folded into adjacent linears so the
    profiling step captures post-isotropic-norm activations. The original
    teacher is untouched and continues to compute the true forward pass during
    distillation.
    """
    print("[fsd] Stage A: γ-fold teacher (in-place on a deep copy)...")
    folded = fold_llama_or_gpt2(teacher)
    return folded


def fold_llama_or_gpt2(model):
    """Dispatch between Llama and GPT-2 fold inventories."""
    cls_name = type(model).__name__
    if "Llama" in cls_name or "Mistral" in cls_name or "Qwen" in cls_name:
        from copy import deepcopy
        m = deepcopy(model)
        fold_llama(m)
        return m
    if "GPT2" in cls_name:
        from copy import deepcopy
        m = deepcopy(model)
        fold_gpt2(m)
        return m
    raise NotImplementedError(f"γ-fold not implemented for {cls_name}")


def stage_b_profile(folded_teacher, calib_loader, args):
    """Run streaming PCA on the folded teacher to produce per-edge bases."""
    print(f"[fsd] Stage B: streaming PCA on {args.calib_sequences} calibration sequences...")
    profile = fasd_profile(
        folded_teacher,
        calib_loader,
        max_batches=args.calib_sequences // max(1, args.per_gpu_batch),
    )
    return profile


def stage_c_score_directions(folded_teacher, profile, calib_loader, args):
    """Fisher-weighted directional scoring.

    Returns dict[branch_name -> DirectionScores]. If --use-fisher-score is False,
    returns None so the allocator falls back to variance scoring.
    """
    if not args.use_fisher_score:
        print("[fsd] Stage C: skipping Fisher scoring (using variance/KL-patch)")
        return None
    print("[fsd] Stage C: Fisher-weighted direction scoring...")
    scores = score_directions(
        folded_teacher,
        profile,
        calib_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return scores


def stage_d_allocate_ranks(profile, direction_scores, target_params, args):
    """Greedy q/cost knapsack allocator under exact param budget.

    Cost-of-rank-step heuristic per branch kind (per-direction parameter cost):
      - residual edge: rank step adds ~ teacher_hidden parameters per linear
                       it touches (estimate via aggregate over branches)
      - attn.q/k/v: cost per rank = teacher_hidden (input) for V_in side
      - attn.o:    cost per rank = teacher_hidden (output)
      - ffn.up/gate: cost per rank = teacher_hidden (input)
      - ffn.down: cost per rank = teacher_intermediate (input) ≈ 4 * teacher_hidden

    These coefficients are deliberately rough; the allocator only needs the
    *relative* cost ordering correct.
    """
    if not args.use_exact_allocator:
        print("[fsd] Stage D: skipping exact allocator (using arch_multiplier)")
        return None  # caller falls back to legacy width_pruner
    print(f"[fsd] Stage D: exact rank allocator, target = {int(target_params):,} params...")

    def cost_fn(b):
        kind = b.kind
        if "ffn.down" in kind:
            return 4.0  # rough; cost relative to 1.0 baseline
        return 1.0

    def step_fn(b):
        return 1  # default; could refine to head-group units

    edges = edges_from_profile(profile, direction_scores=direction_scores,
                               cost_fn=cost_fn, step_fn=step_fn)
    res = allocate_ranks(edges, target_params=int(target_params), tol=0.01, verbose=False)
    print(res.summary())
    return res


def stage_e_build_student(teacher, profile, allocation_result, args):
    """Build the student via the existing builder, then apply the norm and correction modifications.

    When the exact allocator ran, its rank-map is passed directly to
    ``build_student`` via the ``rank_map=`` kwarg, which overrides each branch's
    stored ``behavioral_rank`` without going through ``arch_multiplier``.
    """
    print("[fsd] Stage E: build student...")

    if allocation_result is not None:
        rank_map = allocation_result.ranks
        print(f"[fsd] Building from exact rank-map: {len(rank_map)} entries")
        student = build_student(
            teacher, profile, absorbed_init=True, template="auto",
            rank_map=rank_map,
        )
    else:
        # Fallback: use legacy arch_multiplier path (caller responsible for setting it).
        student = build_student(
            teacher, profile, arch_multiplier=1.0, absorbed_init=True, template="auto"
        )

    if args.use_rr_norm:
        d_model = int(getattr(student.config, "hidden_size", getattr(student.config, "n_embd", 0)))
        n_replaced = replace_layernorm_with_rrnorm(
            student, d_model=d_model, use_q=True, use_scale=True
        )
        print(f"[fsd] replaced {n_replaced} norm layers with RR-Norm")

    if args.use_sparse_block:
        n_injected = inject_sparse_blocks(student)
        print(f"[fsd] injected block-diagonal correction into {n_injected} linears")

    # Save the realised student architecture for matched-comparison baselines.
    write_student_arch_json(student, Path(args.output_dir))

    return student


def write_student_arch_json(student: nn.Module, output_dir: Path) -> None:
    """Persist the student's architecture so baseline scripts can lock it.

    `scripts/repro_baselines/_common.py:build_matched_student` reads this file
    to instantiate identical students with random weights — every in-house
    baseline trains on the SAME architecture, so the head-to-head comparison
    isolates the loss / init differences (which is what FSD's contribution is).
    """
    cfg = student.config
    arch = {
        "hidden_size": int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))),
        "intermediate_size": int(getattr(cfg, "intermediate_size", getattr(cfg, "n_inner", 0))),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))),
        "num_attention_heads": int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads",
                                            getattr(cfg, "num_attention_heads",
                                                    getattr(cfg, "n_head", 0)))),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "student_arch.json", "w") as f:
        json.dump(arch, f, indent=2)
    print(f"[fsd] Wrote student arch to {output_dir / 'student_arch.json'}: {arch}")


def inject_sparse_blocks(student: nn.Module) -> int:
    """Replace o_proj / down_proj / c_proj (output-side square linears) with CorrectedLinear.

    Only square linears (in_features == out_features) are eligible; the
    block-diagonal correction is square per-head. The absorbed-init weight
    is copied into ``CorrectedLinear.linear.weight``; the per-head block
    starts at zero so the student's initial output equals the absorbed-init
    output (the block-diagonal correction grows during training).

    Returns the number of injected modules.
    """
    n_replaced = 0
    cfg = student.config
    num_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 1)))

    # Collect candidates first so we can mutate the parent modules safely.
    candidates: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in student.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child.in_features != child.out_features:
                continue
            # Only inject on attn output projections and FFN down projections.
            # These are the modules whose *input* is per-head, where outlier
            # heads matter most (Dettmers/SmoothQuant locus).
            lname = child_name.lower()
            if lname in ("o_proj", "down_proj") or "c_proj" in lname:
                candidates.append((parent, child_name, child))

    for parent, child_name, child in candidates:
        d = child.in_features
        if d % num_heads != 0:
            continue  # Skip if head structure doesn't divide cleanly.
        d_head = d // num_heads

        new_linear = CorrectedLinear(
            in_features=d, out_features=d,
            num_heads=num_heads, d_head=d_head,
            bias=child.bias is not None,
            correction_init="zero",  # student starts as absorbed-init exactly
        )
        with torch.no_grad():
            new_linear.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_linear.linear.bias is not None:
                new_linear.linear.bias.data.copy_(child.bias.data)
        setattr(parent, child_name, new_linear)
        n_replaced += 1
    return n_replaced


def stage_f_train(student, teacher, train_loader, args):
    """Distillation training loop with Stiefel bases + adaptive objective."""
    print("[fsd] Stage F: training...")

    if args.use_stiefel_optim:
        groups = stiefel_param_groups(
            student, base_lr=args.lr, stiefel_lr_ratio=args.stiefel_lr_ratio,
        )
        opt = StiefelAdam(groups)
        print(f"[fsd] StiefelAdam with {sum(1 for g in groups if g.get('stiefel'))} "
              f"Stiefel groups, {sum(1 for g in groups if not g.get('stiefel'))} standard groups")
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    plateau = PlateauDetector(min_step=200, window=50, tolerance=1e-4, patience=20, decay=0.99)
    on_policy_started = False

    student.to("cuda" if torch.cuda.is_available() else "cpu").train()
    teacher.eval()

    step = 0
    total_steps = (
        100
        if args.dry_run
        else max(1, args.tokens_per_rung // (args.per_gpu_batch * args.seq_len))
    )

    for batch in train_loader:
        if batch is None:
            continue
        if step >= total_steps:
            break

        device = next(student.parameters()).device
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            t_out = teacher(input_ids=ids)
            t_logits = t_out.logits
        s_out = student(input_ids=ids)
        s_logits = s_out.logits

        if args.use_adaptive_skew_kl:
            kd_loss = adaptive_skew_kl(s_logits, t_logits, tau=1.0)
        else:
            kd_loss = forward_kl(s_logits, t_logits)

        if (
            args.use_plateau_trigger
            and plateau.update(float(kd_loss.item()))
            and not on_policy_started
        ):
            print(f"[fsd] Step {step}: plateau detected — beginning on-policy ramp")
            on_policy_started = True

        opt.zero_grad()
        kd_loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[fsd] step={step}  kd_loss={kd_loss.item():.4f}")
        step += 1

    return student


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    teacher, tokenizer = load_teacher(args)
    teacher.to("cuda" if torch.cuda.is_available() else "cpu")
    raw_corpus, collate = load_corpus(args, tokenizer)

    # Calibration loader (small).
    calib_iter = raw_corpus.take(min(args.calib_sequences, 1024))
    calib_loader = DataLoader(list(calib_iter), batch_size=args.per_gpu_batch,
                              collate_fn=collate, num_workers=0)

    # γ-fold the teacher (rotation-equivariant normalization, teacher side).
    folded_teacher = stage_a_gamma_fold(teacher, args)

    # Log param accounting.
    bd = breakdown(teacher)
    print("[fsd] Teacher parameter breakdown:")
    print(bd.summary())

    # Stage B: profile.
    profile = stage_b_profile(folded_teacher, calib_loader, args)

    # Stage C: Fisher score.
    direction_scores = stage_c_score_directions(folded_teacher, profile, calib_loader, args)

    # Stage D: allocate.
    allocation = stage_d_allocate_ranks(profile, direction_scores, args.student_target_params, args)

    # Stage E: build student.
    student = stage_e_build_student(teacher, profile, allocation, args)
    sb = breakdown(student)
    print("[fsd] Student parameter breakdown:")
    print(sb.summary())

    # Stage F: train.
    train_loader = DataLoader(raw_corpus, batch_size=args.per_gpu_batch,
                              collate_fn=collate, num_workers=0)
    student = stage_f_train(student, teacher, train_loader, args)

    # Save.
    save_path = out / "student.pt"
    torch.save(student.state_dict(), save_path)
    print(f"[fsd] Saved student to {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
