#!/usr/bin/env python3
"""End-to-end GPT-2 / WikiText-2 distillation example using the ``asd`` library.

Usage::

    pip install transformers datasets
    python scripts/distill_gpt2_wikitext.py --epochs 3

Writes a narrower GPT-2 student and prints the best validation perplexity.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asd


def get_dataloaders(batch_size: int, seq_len: int):
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    class _WT2(Dataset):
        def __init__(self, split, tokenizer, seq_len):
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [t for t in ds["text"] if t.strip()]
            ids = tokenizer.encode("\n\n".join(texts))
            n = len(ids) // seq_len
            self.tokens = torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)

        def __len__(self):
            return self.tokens.shape[0]

        def __getitem__(self, idx):
            return self.tokens[idx]

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    return {
        "train": DataLoader(_WT2("train", tok, seq_len), batch_size=batch_size,
                            shuffle=True, num_workers=2, drop_last=True,
                            pin_memory=torch.cuda.is_available()),
        "val": DataLoader(_WT2("validation", tok, seq_len), batch_size=batch_size,
                          shuffle=False, num_workers=2,
                          pin_memory=torch.cuda.is_available()),
    }


@torch.no_grad()
def perplexity(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for ids in loader:
        ids = ids.to(device)
        out = model(ids, labels=ids)
        n = ids.shape[0] * (ids.shape[1] - 1)
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="task CE weight")
    ap.add_argument("--beta", type=float, default=1.0, help="subspace weight")
    ap.add_argument("--delta", type=float, default=1.0, help="KD weight")
    ap.add_argument("--kd-T", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--objective", default="cka",
                    choices=["coord_mse", "gram", "cka"])
    ap.add_argument("--source", default="output",
                    choices=["output", "delta"])
    ap.add_argument("--arch-multiplier", type=float, default=1.0)
    ap.add_argument("--profile-batches", type=int, default=20)
    ap.add_argument("--output-dir", default="outputs/distill_gpt2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import GPT2LMHeadModel
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    loaders = get_dataloaders(args.batch_size, args.seq_len)
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params")

    print(f"\nprofiling teacher over {args.profile_batches} batches...")
    t0 = time.time()
    profile = asd.profile(
        teacher, loaders["train"],
        source=args.source, noise_model="mp", n_effective=5000,
        max_batches=args.profile_batches, device=device,
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  per-block effective ranks: {profile.effective_ranks()}")

    # --- student ---
    student = asd.build_student(
        teacher, profile, arch_multiplier=args.arch_multiplier,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"\nstudent: {s_params:,} params ({t_params/s_params:.2f}× compression)")
    print(f"  n_embd: {student.config.n_embd}  (teacher: {teacher.config.n_embd})")

    print("\nteacher ppl...")
    t_ppl = perplexity(teacher, loaders["val"], device)
    print(f"  {t_ppl:.2f}")

    # --- loss ---
    loss_fn = asd.SubspaceLoss(
        profile, objective=args.objective, normalize_features=True,
    ).to(device)

    # Prime projectors so they're in the optimizer.
    ids0 = next(iter(loaders["train"])).to(device)
    with asd.capture(teacher, profile) as t_cap0:
        teacher(ids0)
    with asd.capture(student, profile) as s_cap0:
        student(ids0)
    _ = loss_fn(s_cap0.values(), t_cap0.values())

    params = list(student.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loaders["train"]),
    )

    history = []
    for epoch in range(args.epochs):
        student.train()
        running = {"total": 0.0, "task": 0.0, "sub": 0.0, "kd": 0.0, "n": 0}
        t0 = time.time()
        for ids in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
            ids = ids.to(device)
            with asd.capture(teacher, profile) as t_cap:
                with torch.no_grad():
                    t_out = teacher(ids, labels=ids)
            with asd.capture(student, profile) as s_cap:
                s_out = student(ids, labels=ids)

            task = s_out.loss
            sub = loss_fn(s_cap.values(), t_cap.values())
            T = args.kd_T
            V = s_out.logits.shape[-1]
            kd = F.kl_div(
                F.log_softmax(s_out.logits.reshape(-1, V) / T, dim=-1),
                F.softmax(t_out.logits.reshape(-1, V) / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            total = args.alpha * task + args.beta * sub + args.delta * kd

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            for k, v in (("total", total), ("task", task), ("sub", sub), ("kd", kd)):
                running[k] += float(v.detach().item())
            running["n"] += 1

        val_ppl = perplexity(student, loaders["val"], device)
        avg = {k: v / max(running["n"], 1) for k, v in running.items() if k != "n"}
        print(f"epoch {epoch}: total={avg['total']:.4f} task={avg['task']:.4f} "
              f"sub={avg['sub']:.4f} kd={avg['kd']:.4f} | val_ppl={val_ppl:.2f} | "
              f"{time.time()-t0:.0f}s")
        history.append({**avg, "val_ppl": val_ppl, "epoch": epoch})

    best = min(r["val_ppl"] for r in history)
    import json
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump({
            "teacher_ppl": t_ppl, "best_val_ppl": best,
            "teacher_params": t_params, "student_params": s_params,
            "compression": t_params / s_params,
            "objective": args.objective, "source": args.source,
            "arch_multiplier": args.arch_multiplier,
            "history": history,
        }, f, indent=2)
    profile.save(os.path.join(args.output_dir, "teacher.profile"))

    print(f"\nteacher ppl: {t_ppl:.2f}  best student ppl: {best:.2f}  "
          f"compression: {t_params/s_params:.2f}×")


if __name__ == "__main__":
    main()
