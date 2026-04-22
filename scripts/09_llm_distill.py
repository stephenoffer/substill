#!/usr/bin/env python3
"""ASD distillation for GPT-2 small on WikiText-2.

Applies the same activation-subspace-distillation idea to a decoder-only
transformer:
  1. Capture per-block activation covariance at the output of each transformer
     block (shape (B, T, C); covariance over C using B·T samples per batch).
  2. Per-block effective rank gives the student's hidden size per block.
  3. Build a student GPT-2 with (optionally) fewer layers and smaller hidden
     dim, then distill via CE-on-tokens + logit-KD + per-block subspace loss.

Measures perplexity on the wikitext-2 validation split.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------
# Data
# -------------------------

class WikiText2(Dataset):
    """Pre-tokenized WikiText-2, chunked into fixed-length sequences."""

    def __init__(self, split: str, tokenizer: GPT2Tokenizer, seq_len: int = 256):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [t for t in ds["text"] if t.strip()]
        # Join then tokenize (faster than per-line tokenize + concat)
        all_ids = tokenizer.encode("\n\n".join(texts))
        n_chunks = len(all_ids) // seq_len
        all_ids = all_ids[: n_chunks * seq_len]
        self.tokens = torch.tensor(all_ids, dtype=torch.long).view(n_chunks, seq_len)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        x = self.tokens[idx]
        return x, x  # Autoregressive: target == input (shifted inside model)


def get_dataloaders(batch_size: int, seq_len: int, num_workers: int = 2):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    train = WikiText2("train", tok, seq_len=seq_len)
    val = WikiText2("validation", tok, seq_len=seq_len)
    pin = torch.cuda.is_available()
    return tok, {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin, drop_last=True),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin),
    }


# -------------------------
# Profiling
# -------------------------

class TransformerBlockProfiler:
    """Capture per-block output covariance on GPT-2 in per-token mode.

    Hooks the output of each transformer block in `model.transformer.h[i]`
    (the residual stream, shape (B, T, C)) and accumulates a (C, C) channel
    covariance matrix using B·T samples per batch.
    """

    def __init__(self, model: GPT2LMHeadModel, device: str):
        self.model = model
        self.device = device
        self.n_layers = model.config.n_layer
        self.hidden = model.config.n_embd
        # Double-precision accumulators, on CPU
        self.n = [0] * self.n_layers
        self.sum_x = [torch.zeros(self.hidden, dtype=torch.float64) for _ in range(self.n_layers)]
        self.sum_xx = [torch.zeros(self.hidden, self.hidden, dtype=torch.float64)
                       for _ in range(self.n_layers)]
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, idx: int):
        def fn(module, inputs, output):
            # GPT2 block output is a tuple (hidden_states, ...)
            hs = output[0] if isinstance(output, tuple) else output
            # hs: (B, T, C) → (B*T, C)
            flat = hs.detach().reshape(-1, hs.shape[-1]).to(torch.float64).cpu()
            self.n[idx] += flat.shape[0]
            self.sum_x[idx] += flat.sum(dim=0)
            self.sum_xx[idx] += flat.T @ flat
        return fn

    def register(self):
        for i, block in enumerate(self.model.transformer.h):
            self._hooks.append(block.register_forward_hook(self._hook(i)))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def run(self, loader: DataLoader, max_batches: int | None = None):
        self.model.eval()
        self.register()
        try:
            for i, (ids, _) in enumerate(tqdm(loader, desc="Profiling")):
                if max_batches is not None and i >= max_batches:
                    break
                ids = ids.to(self.device)
                self.model(ids)
        finally:
            self.remove()

    def finalize(self) -> list[dict]:
        """Return list of per-layer dicts: {eigenvalues, components, cov_trace}."""
        out = []
        for i in range(self.n_layers):
            n = max(self.n[i], 1)
            mean = self.sum_x[i] / n
            cov = self.sum_xx[i] / n - mean.unsqueeze(1) * mean.unsqueeze(0)
            cov = cov.float()
            ev, vec = torch.linalg.eigh(cov)
            order = torch.argsort(ev, descending=True)
            ev = ev[order].clamp(min=0)
            vec = vec[:, order]
            out.append({"eigenvalues": ev, "components": vec})
        return out


def effective_rank_variance(eigenvalues: torch.Tensor, threshold: float) -> int:
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1
    cum = torch.cumsum(eigenvalues, dim=0) / total
    mask = cum >= threshold
    if not mask.any():
        return int(len(eigenvalues))
    return int(mask.nonzero()[0].item()) + 1


# -------------------------
# Student construction
# -------------------------

def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def build_student(
    teacher: GPT2LMHeadModel,
    per_layer_rank: list[int],
    student_layers: int | None = None,
    head_multiple: int = 12,
) -> GPT2LMHeadModel:
    """Build a GPT-2 student with reduced hidden size.

    The student uses a single hidden size = max(per_layer_rank) rounded up so
    that the number of attention heads divides it evenly. GPT-2 architecture
    requires `n_embd % n_head == 0`.

    If student_layers is None, keep the same depth as teacher.
    """
    cfg_t = teacher.config
    new_embd = max(per_layer_rank)
    # Round up so that head_multiple divides it (GPT2 n_head=12 → embd % 12 == 0).
    new_embd = round_up(new_embd, head_multiple)
    # Keep tied vocab size (50257)
    cfg_s = GPT2Config(
        vocab_size=cfg_t.vocab_size,
        n_positions=cfg_t.n_positions,
        n_embd=new_embd,
        n_layer=student_layers or cfg_t.n_layer,
        n_head=head_multiple,
        activation_function=cfg_t.activation_function,
        resid_pdrop=cfg_t.resid_pdrop,
        embd_pdrop=cfg_t.embd_pdrop,
        attn_pdrop=cfg_t.attn_pdrop,
        bos_token_id=cfg_t.bos_token_id,
        eos_token_id=cfg_t.eos_token_id,
    )
    return GPT2LMHeadModel(cfg_s)


# -------------------------
# Distillation loss
# -------------------------

class LLMASDLoss(nn.Module):
    """CE + Hinton KD + per-block subspace matching for GPT-2 distillation."""

    def __init__(
        self,
        profiles: list[dict],
        student_ranks: list[int],
        teacher_hidden: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        delta: float = 1.0,
        T: float = 4.0,
        sv_weighting: str = "sqrt",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.T = T
        self.sv_weighting = sv_weighting
        self.n_layers = len(profiles)

        # For each block, store top-k teacher principal components and SV weights
        for i, prof in enumerate(profiles):
            k = student_ranks[i]
            comps = prof["components"][:, :k].clone()  # (C_t, k)
            ev = prof["eigenvalues"][:k].clone()
            self.register_buffer(f"comp_{i}", comps)
            if sv_weighting == "sqrt":
                w = ev.clamp(min=0).sqrt()
                w = w / w.mean().clamp(min=1e-10)
                self.register_buffer(f"svw_{i}", w)
            elif sv_weighting == "uniform":
                pass
            else:
                raise ValueError(f"Unknown sv_weighting: {sv_weighting}")

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, T, V)
        teacher_logits: torch.Tensor,   # (B, T, V)
        student_hiddens: list[torch.Tensor],  # per-block (B, T, C_s)
        teacher_hiddens: list[torch.Tensor],  # per-block (B, T, C_t)
        projectors: nn.ModuleList,             # list of Linear(C_s, k) per block
        labels: torch.Tensor,                   # (B, T)
    ) -> dict[str, torch.Tensor]:
        B, Tlen, V = student_logits.shape

        # Task loss: autoregressive CE — shift so token at position t predicts t+1.
        shift_s = student_logits[:, :-1, :].reshape(-1, V)
        shift_t = labels[:, 1:].reshape(-1)
        loss_task = F.cross_entropy(shift_s, shift_t)

        # Logit KD
        s_log = F.log_softmax(student_logits / self.T, dim=-1)
        t_prob = F.softmax(teacher_logits / self.T, dim=-1)
        loss_kd = F.kl_div(s_log.view(-1, V), t_prob.view(-1, V), reduction="batchmean") * (self.T ** 2)

        # Per-block subspace loss (spatial — per-token)
        loss_sub = torch.zeros((), device=student_logits.device)
        for i, (s_h, t_h, proj) in enumerate(zip(student_hiddens, teacher_hiddens, projectors)):
            comps = getattr(self, f"comp_{i}")  # (C_t, k)
            # Project teacher: (B, T, C_t) @ (C_t, k) = (B, T, k)
            t_proj = t_h @ comps
            # Student projected: (B, T, C_s) → Linear → (B, T, k)
            s_proj = proj(s_h)
            diff_sq = (s_proj - t_proj) ** 2
            if hasattr(self, f"svw_{i}"):
                w = getattr(self, f"svw_{i}")  # (k,)
                loss_sub = loss_sub + (diff_sq * w).mean()
            else:
                loss_sub = loss_sub + diff_sq.mean()
        loss_sub = loss_sub / self.n_layers

        total = self.alpha * loss_task + self.beta * loss_sub + self.delta * loss_kd
        return {"total": total, "task": loss_task.detach(), "subspace": loss_sub.detach(), "kd": loss_kd.detach()}


# -------------------------
# Training
# -------------------------

def forward_with_hidden_states(model: GPT2LMHeadModel, ids: torch.Tensor):
    out = model(ids, output_hidden_states=True)
    # hidden_states: tuple of length n_layer + 1, element [0] is post-embedding,
    # [i] is after block i-1. We want per-block outputs = hidden_states[1..n_layer].
    block_hs = list(out.hidden_states[1:])
    return out.logits, block_hs


@torch.no_grad()
def perplexity(model: GPT2LMHeadModel, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for ids, _ in loader:
        ids = ids.to(device)
        out = model(ids, labels=ids)
        # out.loss is mean CE per token. Multiply by number of tokens used.
        n_tok = (ids.shape[0] * (ids.shape[1] - 1))
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    parser = argparse.ArgumentParser(description="ASD distillation of GPT-2 small on WikiText-2")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--student-layers", type=int, default=None,
                        help="If set, student uses fewer layers than teacher.")
    parser.add_argument("--max-profile-batches", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/llm")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    # Data
    tok, loaders = get_dataloaders(args.batch_size, args.seq_len, num_workers=2)
    tok.pad_token = tok.eos_token
    print(f"Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}")

    # Teacher
    print("Loading GPT-2 teacher...")
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())

    # Profile
    print(f"Profiling {args.max_profile_batches} batches...")
    profiler = TransformerBlockProfiler(teacher, device)
    profiler.run(loaders["train"], max_batches=args.max_profile_batches)
    profiles = profiler.finalize()

    ranks = [effective_rank_variance(p["eigenvalues"], args.threshold) for p in profiles]
    print(f"Per-block effective ranks (τ={args.threshold}): {ranks}")
    print(f"  max: {max(ranks)}, min: {min(ranks)}, teacher_hidden: {teacher.config.n_embd}")

    # Student
    student = build_student(teacher, ranks, student_layers=args.student_layers).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}  ({teacher_params/student_params:.2f}x compression)")
    print(f"Student config: n_layer={student.config.n_layer}, n_embd={student.config.n_embd}")

    # Baseline teacher ppl
    print("\nEvaluating teacher perplexity...")
    t_ppl = perplexity(teacher, loaders["val"], device)
    print(f"  Teacher ppl: {t_ppl:.2f}")

    # Align profile block count to student block count (when student is shallower,
    # pick evenly-spaced teacher blocks to match)
    n_s = student.config.n_layer
    n_t = teacher.config.n_layer
    teacher_block_indices = [int(round(i * (n_t - 1) / max(n_s - 1, 1))) for i in range(n_s)]
    aligned_profiles = [profiles[i] for i in teacher_block_indices]
    aligned_ranks = [ranks[i] for i in teacher_block_indices]
    print(f"  Student depth {n_s}, aligning to teacher blocks {teacher_block_indices}")

    # Projectors: student C_s → k_i per (aligned) block
    projectors = nn.ModuleList([
        nn.Linear(student.config.n_embd, aligned_ranks[i], bias=False)
        for i in range(n_s)
    ]).to(device)
    for p in projectors:
        nn.init.orthogonal_(p.weight)

    # Loss
    loss_fn = LLMASDLoss(
        profiles=aligned_profiles,
        student_ranks=aligned_ranks,
        teacher_hidden=teacher.config.n_embd,
        beta=args.beta, delta=args.delta,
    ).to(device)

    # Optimizer over student + projectors
    params = list(student.parameters()) + list(projectors.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loaders["train"]))

    # Train
    history = []
    for epoch in range(args.epochs):
        student.train()
        projectors.train()
        t0 = time.time()
        running = {"total": 0.0, "task": 0.0, "subspace": 0.0, "kd": 0.0}
        n = 0

        for ids, _ in tqdm(loaders["train"], desc=f"Epoch {epoch}", leave=False):
            ids = ids.to(device)
            with torch.no_grad():
                t_logits, t_hs = forward_with_hidden_states(teacher, ids)

            s_logits, s_hs = forward_with_hidden_states(student, ids)
            # Take the aligned teacher blocks
            t_hs_aligned = [t_hs[i] for i in teacher_block_indices]

            losses = loss_fn(s_logits, t_logits, s_hs, t_hs_aligned, projectors, labels=ids)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            for k in running:
                running[k] += losses[k].item()
            n += 1

        val_ppl = perplexity(student, loaders["val"], device)
        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_total": running["total"] / max(n, 1),
            "train_task": running["task"] / max(n, 1),
            "train_subspace": running["subspace"] / max(n, 1),
            "train_kd": running["kd"] / max(n, 1),
            "val_ppl": val_ppl,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed": elapsed,
        }
        history.append(record)
        print(
            f"Epoch {epoch}: total={record['train_total']:.4f} "
            f"task={record['train_task']:.4f} sub={record['train_subspace']:.4f} "
            f"kd={record['train_kd']:.4f} | val_ppl={val_ppl:.2f} | {elapsed:.0f}s"
        )

    best_ppl = min(r["val_ppl"] for r in history)

    result = {
        "model": "gpt2",
        "dataset": "wikitext-2",
        "threshold": args.threshold,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression": teacher_params / student_params,
        "teacher_ppl": t_ppl,
        "student_ppl_best": best_ppl,
        "student_ppl_final": history[-1]["val_ppl"],
        "student_layers": student.config.n_layer,
        "student_n_embd": student.config.n_embd,
        "teacher_layers": teacher.config.n_layer,
        "teacher_n_embd": teacher.config.n_embd,
        "per_layer_rank": ranks,
        "epochs": args.epochs,
        "history": history,
    }
    with open(os.path.join(args.output_dir, f"result_t{args.threshold:.2f}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Teacher (gpt2): {teacher_params:,} params, ppl={t_ppl:.2f}")
    print(f"Student: {student_params:,} params ({teacher_params/student_params:.2f}x), "
          f"best ppl={best_ppl:.2f} (final {history[-1]['val_ppl']:.2f})")


if __name__ == "__main__":
    main()
