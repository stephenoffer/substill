#!/usr/bin/env python3
"""Debug probe: find which fasd stage hangs on the Anyscale cluster.

Prints a timestamped line before every major step. If the job stalls,
the last-printed stage tells us exactly where.
"""

from __future__ import annotations

import os
import sys
import time

import torch


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    log("importing fasd")
    import fasd  # noqa: E402

    log("loading teacher")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda"
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    log("building synthetic calib loader")
    B, T = 4, 128
    torch.manual_seed(0)
    loader = []
    for _ in range(4):
        ids = torch.randint(100, tok.vocab_size - 100, (B, T), device=device)
        loader.append({
            "input_ids": ids,
            "labels": ids,
            "attention_mask": torch.ones_like(ids),
        })

    log("starting fasd.profile rank_tol=0.02 max_rank=512")
    t0 = time.time()
    profile = fasd.profile(
        teacher,
        loader,
        mode="branch",
        rank_tol=0.02,
        max_rank=512,
        token_weighting="entropy",
        n_calib_batches=4,
        behavioral_calib_batches=2,
        device=device,
    )
    log(f"profile done in {time.time()-t0:.1f}s, {len(profile.branches)} branches")

    log("building absorbed-init student")
    t0 = time.time()
    student = fasd.build_student(
        teacher, profile, absorbed_init=True, template="gpt2"
    ).to(device)
    log(f"build done in {time.time()-t0:.1f}s, params={sum(p.numel() for p in student.parameters())/1e6:.1f}M")

    log("building F_ASDLoss")
    loss_fn = fasd.F_ASDLoss(profile, schedule=fasd.default_schedule()).to(device)

    log("running 5 gram stage steps (frac=0.05)")
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(loss_fn.parameters()), lr=5e-5,
    )
    for s in range(5):
        t_step = time.time()
        batch = loader[s % len(loader)]
        with fasd.capture(teacher, profile, detach=True) as t_hid:
            with torch.no_grad():
                t_out = teacher(**batch)
        t_logits = t_out.logits
        with fasd.capture(student, profile) as s_hid:
            s_out = student(**batch)
        s_logits = s_out.logits
        sub = loss_fn(dict(s_hid.items()), dict(t_hid.items()), step_frac=0.05)
        kd = fasd.forward_kl(s_logits[:, :-1], t_logits[:, :-1])
        total = sub + kd
        opt.zero_grad()
        total.backward()
        opt.step()
        torch.cuda.synchronize()
        log(f"  gram step {s}: {time.time()-t_step:.2f}s sub={sub.item():.3f} kd={kd.item():.3f}")

    log("running 5 procrustes stage steps (frac=0.50)")
    for s in range(5):
        t_step = time.time()
        batch = loader[s % len(loader)]
        with fasd.capture(teacher, profile, detach=True) as t_hid:
            with torch.no_grad():
                t_out = teacher(**batch)
        t_logits = t_out.logits
        with fasd.capture(student, profile) as s_hid:
            s_out = student(**batch)
        s_logits = s_out.logits
        sub = loss_fn(dict(s_hid.items()), dict(t_hid.items()), step_frac=0.50)
        kd = fasd.forward_kl(s_logits[:, :-1], t_logits[:, :-1])
        total = sub + kd
        opt.zero_grad()
        total.backward()
        opt.step()
        torch.cuda.synchronize()
        log(f"  procrustes step {s}: {time.time()-t_step:.2f}s sub={sub.item():.3f} kd={kd.item():.3f}")

    log("DONE")


if __name__ == "__main__":
    main()
