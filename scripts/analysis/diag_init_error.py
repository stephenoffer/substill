"""Diagnostic: where does absorbed-init lose the teacher's function?

Measures, for a GPT-2 student built by the current pipeline:
  * initial validation PPL (no training),
  * per-block relative error between the student's residual stream and the
    teacher's residual stream *projected into the student's basis*,

so we can tell whether the loss is (a) the residual-stream basis, (b) the FFN
intermediate basis, or (c) compounding across depth.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import substill
from substill.builders import _residual_basis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--arch-multiplier", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.float32
    ).eval()
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def chunk(split):
        ids = tok(
            "\n".join(t for t in raw[split]["text"] if t.strip()), return_tensors="pt"
        ).input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [
            {"input_ids": ids[i : i + args.batch_size], "labels": ids[i : i + args.batch_size]}
            for i in range(0, n, args.batch_size)
        ]

    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


@torch.no_grad()
def eval_ppl(model, val, device):
    model.eval().to(device)
    nll = ntok = 0
    for b in val:
        ids = b["input_ids"].to(device)
        lg = model(input_ids=ids).logits[:, :-1].contiguous()
        lab = ids[:, 1:].contiguous()
        nll += float(
            F.cross_entropy(lg.reshape(-1, lg.size(-1)), lab.reshape(-1), reduction="sum")
        )
        ntok += lab.numel()
    return math.exp(min(20, nll / max(1, ntok)))


@torch.no_grad()
def layerwise_drift(teacher, student, V_r, batches, device):
    """Relative error ||h_s - V_r^T h_t|| / ||V_r^T h_t|| per block output."""
    teacher.to(device).eval()
    student.to(device).eval()
    V = V_r.to(device)
    num = None
    den = None
    for b in batches:
        ids = b["input_ids"].to(device)
        t_out = teacher(input_ids=ids, output_hidden_states=True).hidden_states
        s_out = student(input_ids=ids, output_hidden_states=True).hidden_states
        for i, (ht, hs) in enumerate(zip(t_out, s_out, strict=False)):
            proj = ht @ V  # (B,T,k)
            e = (hs - proj).pow(2).sum().item()
            d = proj.pow(2).sum().item()
            if num is None:
                num = [0.0] * len(t_out)
                den = [0.0] * len(t_out)
            num[i] += e
            den[i] += d
    return [math.sqrt(n / max(d, 1e-12)) for n, d in zip(num, den, strict=False)]


def main():
    args = parse_args()
    torch.manual_seed(0)
    teacher, train, val = load(args)
    teacher.to(args.device)
    print(f"teacher params  = {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"teacher PPL     = {eval_ppl(teacher, val, args.device):.2f}")

    calib = train[: args.calib_batches]
    profile = substill.profile(teacher, calib)

    for absorbed in (True, False):
        torch.manual_seed(0)
        student = substill.build_student(
            teacher, profile, arch_multiplier=args.arch_multiplier, absorbed_init=absorbed
        )
        sp = sum(p.numel() for p in student.parameters())
        tag = "absorbed" if absorbed else "random  "
        ppl = eval_ppl(student, val, args.device)
        print(
            f"{tag} init PPL = {ppl:>12.1f}   params={sp:,}  "
            f"ratio={sum(p.numel() for p in teacher.parameters()) / sp:.2f}x  "
            f"n_embd={student.config.n_embd} n_inner={student.config.n_inner}"
        )
        if absorbed:
            V_r = _residual_basis(profile, teacher.config.n_embd, student.config.n_embd)
            drift = layerwise_drift(teacher, student, V_r, val[:4], args.device)
            print("  per-block relative drift ||h_s - V^T h_t|| / ||V^T h_t||:")
            for i, d in enumerate(drift):
                print(f"    block {i:>2}: {d:.4f}")

    # How much residual-stream energy does the channel-selection basis retain?
    for b in profile.branches:
        if getattr(b, "kind", None) == "block.residual":
            ev = b.eigenvalues.float()
            k = int(args.arch_multiplier * teacher.config.n_embd)
            print(
                f"  {b.name}: PCA energy retained @k={k}: "
                f"{ev[:k].sum() / ev.sum():.4f}   (rank={b.behavioral_rank})"
            )
            break


if __name__ == "__main__":
    main()
