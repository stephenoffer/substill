"""Does the GPT-2 residual-basis inversion survive on an RMSNorm, untied-embedding model?

`docs/init_findings.md` §2, on GPT-2:

  * a PCA rotation of the residual stream **diverges** (initial PPL > 1e16),
  * variance-ranked coordinate selection **loses** to plain first-k truncation
    (180.6 vs 161.0 final PPL),
  * and the ordering is *inverted* against retained variance and against logit error.

The mechanism proposed there blames two GPT-2 properties: LayerNorm centers across
coordinates (and one coordinate carries 73.7% of the residual variance), and `lm_head` is
tied to `wte` (which blocks the mean-removal fold that would make rotations legitimate).

`JackFram/llama-160m` has GPT-2's exact shape -- hidden 768, 12 layers, 12 heads,
head_dim 64 -- with **RMSNorm** (no centering) and **untied embeddings**. So the mechanism
predicts: rotations stop diverging, and the inversion weakens or disappears.

If instead identity truncation still wins, §2's explanation is wrong and the phenomenon is
something more general. Either outcome is worth having; the point is that it is decidable.

Every arm shares the student architecture, the gamma fold, the RMS gain correction, the
intermediate basis, the optimizer, the data order, and the seed. Only the residual basis
`V` differs.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import torch
from torch import Tensor  # noqa: F401  (used in an annotation below)


def _disable_triton_overrides() -> None:
    """Llama's rotary embedding builds `inv_freq @ position_ids` as an outer product,
    which PyTorch routes to a Triton `bmm_outer_product` kernel. Triton JIT-compiles at
    runtime and needs a C compiler, which this box does not have. Dropping the override
    falls back to the eager matmul -- same numerics, no codegen."""
    try:
        from torch._native import registry
        registry.deregister_op_overrides(disable_dsl_names="triton")
    except Exception:  # noqa: BLE001 -- older torch, or no override registry
        pass


_disable_triton_overrides()

from scripts.analysis.bench import distill  # noqa: E402
from substill.compression.llama_absorb import (  # noqa: E402
    absorb_llama,
    build_narrow_llama,
    gamma_fold_llama,
    llama_logit_metric,
    llama_residual_second_moment,
    rms_gain,
)
from substill.compression.seq_absorb import (  # noqa: E402
    eval_ppl,
    grassmann_logit_basis,
    relative_logit_error,
    residual_basis,
)


def load(args):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).eval()
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def chunk(split):
        ids = tok("\n".join(t for t in raw[split]["text"] if t.strip()),
                  return_tensors="pt").input_ids[0]
        n = ids.numel() // args.seq_len
        ids = ids[: n * args.seq_len].view(n, args.seq_len)
        return [{"input_ids": ids[i:i + args.batch_size]}
                for i in range(0, n, args.batch_size)]

    return teacher, chunk("train"), chunk("validation")[: args.eval_batches]


@torch.no_grad()
def interm_bases(teacher, calib, interm, device):
    """Per-layer selection of the highest-energy FFN neurons.

    A neuron is one coordinate of `silu(gate) * up`, i.e. one column of `down_proj`.
    SiLU is elementwise, so selecting coordinates commutes with it exactly.
    """
    L = teacher.config.num_hidden_layers
    d_int = teacher.config.intermediate_size
    acc = [torch.zeros(d_int, dtype=torch.float64, device=device) for _ in range(L)]
    n = 0

    def mk(li):
        def hook(_m, inp, _o):
            acc[li] += (inp[0].detach().double() ** 2).reshape(-1, d_int).sum(0)
        return hook

    hooks = [layer.mlp.down_proj.register_forward_hook(mk(li))
             for li, layer in enumerate(teacher.model.layers)]
    for b in calib:
        teacher(input_ids=b["input_ids"].to(device))
        n += b["input_ids"].numel()
    for h in hooks:
        h.remove()

    out = []
    for li in range(L):
        if interm >= d_int:
            out.append(torch.eye(d_int, device=device))
            continue
        top = torch.argsort(acc[li], descending=True)[:interm]
        E = torch.zeros(d_int, interm, device=device)
        E[top, torch.arange(interm, device=device)] = 1.0
        out.append(E)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="JackFram/llama-160m")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--interm", type=int, default=1536)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-kv", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--bases", nargs="+", default=["identity", "random_sel", "select", "pca"])
    p.add_argument("--grassmann-steps", type=int, default=1200)
    p.add_argument("--output", default="runs/llama_basis.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    teacher, train, val = load(args)
    teacher.to(args.device)
    t_ppl = eval_ppl(teacher, val, args.device)
    t_params = sum(p.numel() for p in teacher.parameters())
    c = teacher.config
    print(f"teacher {args.teacher}: PPL={t_ppl:.2f}  params={t_params:,}  "
          f"hidden={c.hidden_size} layers={c.num_hidden_layers} heads={c.num_attention_heads} "
          f"head_dim={c.hidden_size // c.num_attention_heads}  norm=RMSNorm  "
          f"tied={c.tie_word_embeddings}\n", flush=True)

    folded = gamma_fold_llama(teacher).to(args.device).eval()
    calib = train[: args.calib_batches]
    S = llama_residual_second_moment(folded, calib, device=args.device)
    E = interm_bases(folded, calib, args.interm, args.device)

    # energy of the dominant residual coordinate -- the GPT-2 number to compare against
    dg = S.diagonal()
    print(f"residual stream: top-1 coordinate = {dg.max() / dg.sum() * 100:.1f}% of variance, "
          f"max/median = {dg.max() / dg.median():.0f}x   (GPT-2: 73.7%, 9136x)\n", flush=True)

    M = llama_logit_metric(folded).to(args.device)

    rows = []
    for basis in args.bases:
        if basis == "grassmann":
            V = grassmann_logit_basis(S, M, args.hidden, steps=args.grassmann_steps)
        else:
            # `gn`/`select_gn` need the logit (influence) metric M -- this is the
            # activation+influence basis that represents AIR-family compressors.
            V = residual_basis(S, args.hidden, method=basis, M=M).to(args.device)
        V = V.to(args.device)
        gain = rms_gain(S, V)
        retained = float(torch.trace(V.T @ S @ V) / torch.trace(S))
        logit_err = relative_logit_error(S, M, V)
        for seed in args.seeds:
            args.seed = seed
            torch.manual_seed(seed)
            st = build_narrow_llama(folded, args.hidden, args.interm,
                                    args.n_head, args.n_kv).to(args.device)
            absorb_llama(folded, st, V, E, norm_gain=gain)
            sp = sum(p.numel() for p in st.parameters())
            ip = eval_ppl(st.eval(), val, args.device)
            for p in st.parameters():
                p.requires_grad_(True)
            t0 = time.time()
            torch.manual_seed(seed)
            distill(folded, st, train, args, args.device, kd="forward_kl",
                    feat=0.0, V=None, Mk=None, steps=args.steps)
            fp = eval_ppl(st, val, args.device)
            print(f"basis={basis:<11} seed={seed}  var={retained:.3f} logit_err={logit_err:.4f} "
                  f"params={sp/1e6:.1f}M ({t_params/sp:.2f}x)  init={ip:>12,.1f}  "
                  f"final={fp:>8.2f}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"basis": basis, "seed": seed, "params": sp, "var_retained": retained,
                         "logit_err": logit_err, "rms_gain": gain,
                         "init_ppl": ip, "final_ppl": fp})
            del st
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"teacher": args.teacher, "teacher_ppl": t_ppl, "teacher_params": t_params,
         "top1_variance_share": float(dg.max() / dg.sum()), "args": vars(args),
         "rows": rows}, indent=2))

    print("\n" + "=" * 76)
    ppl_hdr = f"final PPL (n={len(args.seeds)})"
    print(f"{'residual basis':<13}{'var kept':>10}{'logit err':>11}{'init PPL':>15}"
          f"{ppl_hdr:>23}")
    print("-" * 76)
    for basis in args.bases:
        v = [r["final_ppl"] for r in rows if r["basis"] == basis]
        i = [r["init_ppl"] for r in rows if r["basis"] == basis]
        vr = [r["var_retained"] for r in rows if r["basis"] == basis][0]
        le = [r["logit_err"] for r in rows if r["basis"] == basis][0]
        sd = stats.stdev(v) if len(v) > 1 else 0.0
        print(f"{basis:<13}{vr:>10.3f}{le:>11.4f}{stats.mean(i):>15,.0f}"
              f"{stats.mean(v):>16.2f} +/- {sd:<5.2f}")
    print(f"\nteacher PPL={t_ppl:.2f}\nwrote {args.output}")


if __name__ == "__main__":
    main()
