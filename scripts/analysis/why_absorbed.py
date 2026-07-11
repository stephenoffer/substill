"""Why does absorbed init work, when it approximates the teacher so badly?

Absorbed init reaches 161.0 PPL after distillation while starting at 7e7 -- worse
than random init, which reaches only 438.9. Its advantage therefore cannot be that
it approximates the teacher's function. What is it carrying?

Three ablations that each destroy one property while keeping the others:

absorbed        teacher submatrices. Function ~destroyed by truncation, but the
                weight matrices keep the teacher's singular values, per-matrix
                scale, entry distribution, and row/column correlations.
permuted        the same matrices with rows and columns independently permuted.
                Singular values, per-matrix scale, and the entry multiset are all
                preserved *exactly*; the function and every cross-layer alignment
                are destroyed.
scaled_random   fresh Gaussian weights, per-matrix std matched to `absorbed`.
                Only the scale survives. (Spectra become Marchenko-Pastur.)
spectral        fresh random orthogonal factors carrying `absorbed`'s exact singular
                values (U R V^T with U, V random orthonormal). Spectrum and scale
                survive; entries, function, and alignment do not.
random          stock init. Nothing survives.

If `scaled_random` lands near `absorbed`, absorbed init contributes nothing beyond
per-matrix scale and the whole subspace-projection apparatus is measuring layer-wise
weight scaling. If it lands near `random`, absorbed init carries real structure and
the interesting question is which of spectrum / entries / alignment supplies it.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import torch

from scripts.analysis.bench import distill
from scripts.analysis.h2h import load
from substill.compression.seq_absorb import (
    _ffn_basis,
    absorb_gpt2,
    build_narrow_gpt2,
    eval_ppl,
    logit_metric,
    residual_basis,
    residual_second_moment,
)

# Only the block weight matrices are ablated. Embeddings (tied to lm_head) and
# LayerNorm affines are left as absorbed init produced them in every arm, so the
# comparison isolates the transformer weights rather than the unembedding.
_MATS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def _block_mats(student):
    for blk in student.transformer.h:
        for name in _MATS:
            mod = blk
            for part in name.split("."):
                mod = getattr(mod, part)
            yield mod


@torch.no_grad()
def ablate(student, mode, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for mod in _block_mats(student):
        W = mod.weight.data
        cpu = W.detach().float().cpu()
        if mode == "permuted":
            r = torch.randperm(cpu.shape[0], generator=g)
            c = torch.randperm(cpu.shape[1], generator=g)
            new = cpu[r][:, c]
        elif mode == "scaled_random":
            new = torch.randn(cpu.shape, generator=g) * cpu.std()
        elif mode == "spectral":
            s = torch.linalg.svdvals(cpu)
            m, n = cpu.shape
            U, _ = torch.linalg.qr(torch.randn(m, min(m, n), generator=g))
            Vt, _ = torch.linalg.qr(torch.randn(n, min(m, n), generator=g))
            new = U @ torch.diag(s) @ Vt.T
        else:
            raise ValueError(mode)
        W.copy_(new.to(W.dtype).to(W.device))
    return student


def build(arm, teacher, calib, args, device):
    if arm == "random":
        torch.manual_seed(args.seed)
        return build_narrow_gpt2(teacher, args.k, args.inner).to(device)
    S = residual_second_moment(teacher, calib, device=device)
    V = residual_basis(S, args.k, method="identity",
                       M=logit_metric(teacher).to(device)).to(device)
    ffn = [_ffn_basis(teacher, i, calib, args.inner, device)
           for i in range(teacher.config.n_layer)]
    st = build_narrow_gpt2(teacher, args.k, args.inner).to(device)
    absorb_gpt2(teacher, st, V, ffn)
    if arm != "absorbed":
        ablate(st, arm, args.seed)
    return st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--k", type=int, default=324)
    p.add_argument("--inner", type=int, default=1068)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=1999)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--arms", nargs="+",
                   default=["absorbed", "permuted", "spectral", "scaled_random", "random"])
    p.add_argument("--output", default="runs/why_absorbed.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    calib = train[: args.calib_batches]
    print(f"teacher PPL={t_ppl:.2f}  steps={args.steps}  lr={args.lr:.0e}\n", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            args.seed = seed
            torch.manual_seed(seed)
            t0 = time.time()
            st = build(arm, teacher, calib, args, args.device)
            ip = eval_ppl(st, val, args.device)
            for p in st.parameters():
                p.requires_grad_(True)
            torch.manual_seed(seed)
            distill(teacher, st, train, args, args.device, kd="forward_kl",
                    feat=0.0, V=None, Mk=None, steps=args.steps)
            fp = eval_ppl(st, val, args.device)
            print(f"{arm:<15} seed={seed}  init={ip:>13,.1f}  final={fp:>8.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            rows.append({"arm": arm, "seed": seed, "init_ppl": ip, "final_ppl": fp})
            del st
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher_ppl": t_ppl, "args": vars(args), "rows": rows}, indent=2))
    print("\n" + "=" * 62)
    print(f"{'arm':<16}{'final PPL':>24}")
    print("-" * 62)
    for arm in args.arms:
        v = [r["final_ppl"] for r in rows if r["arm"] == arm]
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        print(f"{arm:<16}{stats.mean(v):>14.2f} +/- {sd:<6.2f}")
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
