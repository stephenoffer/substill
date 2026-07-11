"""FSD headline experiment on GPT-2 + WikiText-2.

Goal: produce **actual training numbers** comparing three variants at matched
compression and matched compute:

  - **r0_random**:  random-init student, vanilla CE on WikiText-2 (no teacher KD)
  - **r1_kd_random**: random-init student, forward-KL distillation
  - **r2_fasd**:    absorbed-init student (existing F-ASD path), forward-KL
  - **r3_fsd**:     absorbed-init + RR-Norm + sparse-block correction
                    + adaptive entropy-gap skew-KL

All four train the same student architecture for the same number of steps with
the same optimizer hyperparameters and the same data shuffle. Reports final
WikiText-2 validation PPL per variant.

Usage::

    python scripts/fsd_headline_experiment.py --output runs/headline.json --steps 300

This is the smoke-test version: GPT-2-small (124M) → ~85M student, WikiText-2 raw,
~5-10 minutes on a single A10G. For the *real* paper headline (Llama-3.2-3B → 1B
on SlimPajama), see ``scripts/distill_llama32_fsd.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from substill.api import profile as fasd_profile
from substill.builders import build_student
from substill.compression.diff_rank import DifferentiableRankGate, RankBudgetController
from substill.compression.factored_linear import TeacherFactoredLinear
from substill.losses.generative_kd import adaptive_skew_kl, forward_kl
from substill.pipeline import convert_gpt2_to_factored
from substill.profiling.gamma_fold import fold_gpt2
from substill.util.param_accounting import count_params
from substill.util.rr_norm import replace_layernorm_with_rrnorm

# CPSD variant set (the novel cells) appended to the legacy FSD variants.
CPSD_VARIANTS = ("cpsd_mt", "cpsd_full")


# ---- experiment setup ------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2",
                   help="HF model id (default gpt2 = 124M). Try gpt2-medium for 354M.")
    p.add_argument("--target-compression", type=float, default=1.5,
                   help="Teacher / student parameter ratio.")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", type=str,
                   default="r0_random,r1_kd_random,r2_fasd,r3_fsd",
                   help="Comma-separated list of variants to run.")
    p.add_argument("--output", type=str, default="runs/headline.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher_and_data(args):
    """Load teacher and tokenize WikiText-2.

    ``--teacher tiny`` builds a small random GPT-2 + synthetic data for an offline
    smoke test of the full pipeline (no HF download); used to validate the CPSD
    variants before submitting real Anyscale jobs.
    """
    if args.teacher == "tiny":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=128, n_positions=args.seq_len, n_embd=64,
                         n_layer=2, n_head=4, n_inner=256)
        cfg.pad_token_id = 0
        teacher = GPT2LMHeadModel(cfg).eval()
        torch.manual_seed(0)
        train_ids = torch.randint(5, 120, (64, args.seq_len))
        val_ids = torch.randint(5, 120, (16, args.seq_len))
        print(f"[exp] TINY offline teacher: {count_params(teacher)/1e6:.2f}M params")
        return teacher, None, train_ids, val_ids

    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[exp] Loading teacher {args.teacher}...")
    tok = GPT2Tokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = GPT2LMHeadModel.from_pretrained(args.teacher).eval()

    print("[exp] Loading WikiText-2...")
    # Newer huggingface_hub requires a namespaced repo id ("namespace/name"); the bare
    # "wikitext" script id builds an invalid hf:// URI. Try the canonical namespaced
    # repo first, fall back to the legacy id for older datasets versions.
    raw = None
    for ds_id in ("Salesforce/wikitext", "wikitext"):
        try:
            raw = load_dataset(ds_id, "wikitext-2-raw-v1")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[exp]   load_dataset({ds_id!r}) failed: {type(e).__name__}: {e}")
    if raw is None:
        raise RuntimeError("could not load WikiText-2 under any known repo id")

    def tokenize(ds_split):
        # Concatenate all non-empty lines, then chunk to seq_len.
        text = "\n".join(t for t in ds_split["text"] if t.strip())
        ids = tok(text, return_tensors="pt").input_ids[0]
        n_chunks = ids.numel() // args.seq_len
        ids = ids[: n_chunks * args.seq_len].view(n_chunks, args.seq_len)
        return ids

    train_ids = tokenize(raw["train"])
    val_ids = tokenize(raw["validation"])
    print(f"[exp] train chunks: {train_ids.shape}, val chunks: {val_ids.shape}")
    return teacher, tok, train_ids, val_ids


def make_loader(ids, batch_size, shuffle, seed=0):
    """Tiny custom loader: returns dicts with input_ids."""
    n = ids.shape[0]
    indices = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        if not batch_idx:
            continue
        batch = ids[batch_idx]
        yield {"input_ids": batch}


def calib_batches(train_ids, args):
    return list(make_loader(train_ids, args.batch_size, shuffle=False))[: args.calib_batches]


@torch.no_grad()
def eval_ppl(model, val_ids, args):
    """Compute average per-token cross-entropy on validation set."""
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for batch in list(make_loader(val_ids, args.batch_size, shuffle=False))[: args.eval_batches]:
        ids = batch["input_ids"].to(device)
        out = model(input_ids=ids).logits  # (B, T, V)
        # next-token CE: predict ids[:, 1:] from logits[:, :-1, :]
        shift_logits = out[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(shift_labels.numel())
    avg_nll = total_nll / max(1, total_tokens)
    return math.exp(avg_nll), avg_nll


def build_random_student(teacher, profile, args):
    """Same architecture as F-ASD student, but with random init."""
    # Use the same config the F-ASD path would produce.
    fasd_student = build_student(
        teacher, profile, arch_multiplier=_arch_mult_for_compression(args.target_compression),
        absorbed_init=False, template="gpt2",
    )
    # GPT2LMHeadModel(config) gives random init.
    return fasd_student  # absorbed_init=False already gives random init


def _arch_mult_for_compression(ratio: float) -> float:
    """Heuristic: map compression ratio to arch_multiplier.

    Empirically, arch_multiplier ≈ 1/sqrt(ratio) gives roughly the right
    number of params after width pruning. For ratio=2 → 0.71; ratio=4 → 0.50.
    """
    return float(1.0 / math.sqrt(max(1.0, ratio)))


def build_fasd_student(teacher, profile, args):
    """Existing F-ASD path: absorbed init via build_student, no other pillars."""
    return build_student(
        teacher, profile,
        arch_multiplier=_arch_mult_for_compression(args.target_compression),
        absorbed_init=True, template="gpt2",
    )


def build_fsd_student(teacher, train_ids, args):
    """FSD path: γ-fold(teacher) → re-profile → absorbed init → RR-Norm replacement.

    Rotation-equivariant normalization requires the teacher's LayerNorm γ/β to be
    folded into the next linear *before* profiling. After folding, the teacher's LN modules
    have γ=1, β=0 (parameter-free isotropic normalisation), so:

      - The PCA basis V_r captured from the γ-folded teacher's residual stream
        is the right basis for absorbed init (post-isotropic-norm activations).
      - Absorbed init's `_copy_layernorm` step copies γ=1, β=0 into the student.
      - Replacing the student's LN with RR-Norm is then mathematically exact:
        RR-Norm with scale=1, Q=I, center=True equals LN(γ=1, β=0).

    Without γ-fold first, the student inherits non-trivial γ/β and the RR-Norm
    swap loses them — that's the v9 5-14 OOM init disaster.
    """
    folded_teacher = copy.deepcopy(teacher)
    fold_gpt2(folded_teacher)
    folded_teacher.to(args.device).eval()

    # Re-profile from the γ-folded teacher (post-isotropic-norm activations).
    print("[exp] FSD: re-profiling from γ-folded teacher...")
    folded_calib = calib_batches(train_ids, args)
    folded_profile = fasd_profile(
        folded_teacher, folded_calib,
        n_calib_batches=args.calib_batches,
        behavioral_calib_batches=min(4, args.calib_batches),
    )

    student = build_student(
        folded_teacher, folded_profile,
        arch_multiplier=_arch_mult_for_compression(args.target_compression),
        absorbed_init=True, template="gpt2",
    )
    d_model = int(student.config.n_embd)
    n_replaced = replace_layernorm_with_rrnorm(
        student, d_model=d_model, use_q=True, use_scale=True, center_for_layernorm=True,
    )
    print(f"[exp] FSD: replaced {n_replaced} norm layers with RR-Norm")
    return student, folded_teacher


def count_folded_params(model) -> int:
    """Inference-time param count: TeacherFactoredLinear/_GatedFactored fold their
    Stiefel factors back to a collapsed W_S, so count the folded size, not the
    (larger) training-time factored params. The frozen teacher W_T is a buffer and
    never counted; RR-Norm Q and other real params count normally.
    """
    total = 0
    counted = set()
    for mod in model.modules():
        if isinstance(mod, _GatedFactored):
            mod = mod.tfl
        if isinstance(mod, TeacherFactoredLinear):
            total += mod.k_in * mod.k_out + (mod.k_out if mod.b_T is not None else 0)
            counted.add(id(mod.V_in))
            counted.add(id(mod.V_out))
            if mod.B_free is not None:  # folds into the effective weight; don't double-count
                counted.add(id(mod.B_free))
            for _n, b in mod.named_buffers():
                counted.add(id(b))
    for p in model.parameters():
        if id(p) not in counted:
            total += p.numel()
    return total


class _GatedFactored(nn.Module):
    """TeacherFactoredLinear with a differentiable rank gate on its input latent (DDR)."""

    def __init__(self, tfl: TeacherFactoredLinear):
        super().__init__()
        self.tfl = tfl
        self.gate = DifferentiableRankGate(tfl.k_in, init_open=True)

    def forward(self, x):
        return self.tfl(self.gate(x))


def build_cpsd_student(teacher, train_ids, args, *, use_ddr: bool):
    """CPSD student: γ-fold → re-profile → absorbed (circuit-preserving) init →
    RR-Norm → convert absorbed linears to TeacherFactoredLinear (Stiefel-trainable
    factors). With ``use_ddr``, wrap each factored module in a differentiable rank
    gate and return a budget controller.

    Returns (student, folded_teacher, ddr_controller_or_None).
    """
    folded_teacher = copy.deepcopy(teacher)
    fold_gpt2(folded_teacher)
    folded_teacher.to(args.device).eval()

    folded_calib = calib_batches(train_ids, args)
    folded_profile = fasd_profile(
        folded_teacher, folded_calib,
        n_calib_batches=args.calib_batches,
        behavioral_calib_batches=min(4, args.calib_batches),
    )
    student = build_student(
        folded_teacher, folded_profile,
        arch_multiplier=_arch_mult_for_compression(args.target_compression),
        absorbed_init=True, template="gpt2",
    )
    d_model = int(student.config.n_embd)
    replace_layernorm_with_rrnorm(
        student, d_model=d_model, use_q=True, use_scale=True, center_for_layernorm=True,
    )
    # MT: convert absorbed linears to Stiefel-trainable factored modules. free_core
    # adds a zero-init Euclidean correction so the module has full fitting capacity
    # (basis rotation alone is too few DOF); preserves exact-at-init + compression.
    n_conv = convert_gpt2_to_factored(
        student, folded_teacher, folded_profile, free_core=True)
    print(f"[exp] CPSD: converted {n_conv} absorbed linears to TeacherFactoredLinear")

    controller = None
    if use_ddr:
        gates, costs = {}, {}
        # Wrap each TeacherFactoredLinear in a gated module (walk parents).
        for name, mod in list(student.named_modules()):
            if isinstance(mod, TeacherFactoredLinear):
                gf = _GatedFactored(mod)
                _set_by_name(student, name, gf)
                gates[name] = gf.gate
                costs[name] = torch.full((mod.k_in,), float(mod.d_in + mod.d_out))
        total_full = sum(float(c.sum()) for c in costs.values())
        controller = RankBudgetController(
            gates, costs, target_params=0.7 * total_full, lam=1.0,
        )
        print(f"[exp] CPSD-DDR: {len(gates)} rank gates, budget {0.7*total_full:.0f}")
    return student, folded_teacher, controller


def _set_by_name(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def variant_loss_fn(variant: str):
    if variant in ("r0_random",):
        # Vanilla CE only.
        def f(s_logits, t_logits, ids):
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return f
    if variant in ("r1_kd_random", "r2_fasd", "r3_fsd_kd", "r3_fsd_kd_stiefel"):
        def f(s_logits, t_logits, ids):
            return forward_kl(s_logits, t_logits)
        return f
    if variant in ("r3_fsd", "r3_fsd_full"):
        def f(s_logits, t_logits, ids):
            return adaptive_skew_kl(s_logits, t_logits, tau=1.0)
        return f
    if variant in CPSD_VARIANTS:
        # Match the FSD baseline's KD loss (forward-KL) so the ONLY difference is the
        # novel component (manifold-trained factors / differentiable rank), not the
        # objective. (skew-KL is a separate, known-weaker-at-low-steps axis.)
        def f(s_logits, t_logits, ids):
            return forward_kl(s_logits, t_logits)
        return f
    raise ValueError(f"unknown variant: {variant}")


def train_variant(variant, teacher, train_ids, val_ids, profile, args):
    """Train one variant from scratch and return final metrics."""
    print(f"\n[exp] === {variant} ===")
    set_seed(args.seed)
    device = args.device

    # Build student.
    teacher_for_kd = teacher  # default: use the original teacher for KD supervision
    use_stiefel = False
    ddr_ctrl = None
    if variant in ("r0_random", "r1_kd_random"):
        student = build_random_student(teacher, profile, args)
    elif variant == "r2_fasd":
        student = build_fasd_student(teacher, profile, args)
    elif variant in ("r3_fsd", "r3_fsd_kd", "r3_fsd_kd_stiefel", "r3_fsd_full"):
        student, folded = build_fsd_student(teacher, train_ids, args)
        teacher_for_kd = folded
        # Use StiefelAdam to keep RR-Norm's Q matrices on the manifold.
        if variant in ("r3_fsd_kd_stiefel", "r3_fsd_full"):
            use_stiefel = True
    elif variant in CPSD_VARIANTS:
        # CPSD: circuit-preserving init + Stiefel-trained projection factors
        # (+ differentiable rank for cpsd_full). Trains V_in/V_out AND RR-Norm Q.
        student, folded, ddr_ctrl = build_cpsd_student(
            teacher, train_ids, args, use_ddr=(variant == "cpsd_full"))
        teacher_for_kd = folded
        use_stiefel = True
    else:
        raise ValueError(variant)

    student.to(device).train()
    teacher.to(device).eval()

    n_teacher = count_params(teacher)
    # For CPSD, report the INFERENCE (folded) param count — factored V_in/V_out
    # collapse to W_S at deployment, so the training-time count overstates size.
    n_student = count_folded_params(student) if variant in CPSD_VARIANTS \
        else count_params(student)
    n_train = count_params(student)
    extra = f" (train {n_train/1e6:.1f}M)" if n_train != n_student else ""
    print(f"[exp] {variant}: student {n_student/1e6:.1f}M params{extra}, "
          f"teacher {n_teacher/1e6:.1f}M, ratio {n_teacher/n_student:.2f}x")

    # Initial PPL.
    init_ppl, init_nll = eval_ppl(student, val_ids, args)
    print(f"[exp] {variant}: init val PPL = {init_ppl:.3e}  (nll={init_nll:.3f})")

    # Training.
    if use_stiefel:
        from substill.training.stiefel_optim import StiefelAdam, stiefel_param_groups
        groups = stiefel_param_groups(
            student, base_lr=args.lr, stiefel_lr_ratio=0.1, weight_decay=0.01,
        )
        opt = StiefelAdam(groups)
        n_st = sum(1 for g in groups if g.get("stiefel"))
        print(f"[exp] {variant}: StiefelAdam ({n_st} Stiefel groups)")
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = variant_loss_fn(variant)

    losses: list[float] = []
    t0 = time.time()
    step = 0
    while step < args.steps:
        for batch in make_loader(train_ids, args.batch_size, shuffle=True, seed=args.seed + step):
            if step >= args.steps:
                break
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                t_logits = teacher_for_kd(input_ids=ids).logits
            s_logits = student(input_ids=ids).logits
            loss = loss_fn(s_logits, t_logits, ids)
            if ddr_ctrl is not None:
                loss = loss + ddr_ctrl.budget_penalty()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0 or step == args.steps:
                elapsed = time.time() - t0
                print(f"[exp] {variant} step {step}/{args.steps}  "
                      f"loss={loss.item():.4f}  ({elapsed:.1f}s)")

    # Final PPL.
    final_ppl, final_nll = eval_ppl(student, val_ids, args)
    elapsed = time.time() - t0
    print(f"[exp] {variant}: final val PPL = {final_ppl:.2f}  (nll={final_nll:.3f})  "
          f"[{elapsed:.1f}s]")

    # Cleanup to free GPU memory before next variant.
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "student_params": int(n_student),
        "train_params": int(n_train),
        "compression_ratio": float(n_teacher / n_student),
        "init_ppl": float(init_ppl),
        "final_ppl": float(final_ppl),
        "init_nll": float(init_nll),
        "final_nll": float(final_nll),
        "wall_seconds": float(elapsed),
        "loss_curve": losses,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    teacher, tokenizer, train_ids, val_ids = load_teacher_and_data(args)

    print("[exp] Profiling teacher (streaming PCA)...")
    calib = calib_batches(train_ids, args)
    teacher.to(args.device)
    profile = fasd_profile(
        teacher, calib,
        n_calib_batches=args.calib_batches,
        behavioral_calib_batches=min(4, args.calib_batches),
    )
    print(f"[exp] Profile: {len(profile.branches)} branches")

    # Teacher PPL (sanity check).
    t_ppl, t_nll = eval_ppl(teacher, val_ids, args)
    print(f"[exp] Teacher val PPL = {t_ppl:.2f}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = []
    for variant in variants:
        try:
            res = train_variant(variant, teacher, train_ids, val_ids, profile, args)
            results.append(res)
        except Exception as e:
            print(f"[exp] {variant} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"variant": variant, "status": "failed", "error": str(e)})

    # Output.
    summary = {
        "teacher": args.teacher,
        "teacher_params": int(count_params(teacher)),
        "teacher_ppl": float(t_ppl),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "target_compression": args.target_compression,
        "seed": args.seed,
        "device": str(args.device),
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[exp] Wrote {out}")
    print("\n=== SUMMARY ===")
    print(f"Teacher PPL: {t_ppl:.2f}")
    for r in results:
        if r.get("status") == "failed":
            print(f"  {r['variant']:15s} FAILED: {r.get('error', '?')}")
        else:
            print(f"  {r['variant']:15s} {r['student_params']/1e6:5.1f}M  "
                  f"init={r['init_ppl']:.2e}  final={r['final_ppl']:7.2f}  "
                  f"({r['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
