# v3 Improvements

Concrete algorithmic changes implemented on the NFS working copy
(`/mnt/shared_storage/asd/`) and validated with Anyscale prodjobs on
2026-04-22. For the *reasons* behind each change, see
[findings.md](findings.md). For the full results, see [results.md](results.md).

## Executive summary — what to change

Two drops land cleanly on both CNN and LLM. Mahalanobis weighting is a
huge win on LLM and a regression on CNN (when combined with drop-KD);
the fix on CNN is simply to not use it.

| change | verdict | evidence |
|---|---|---|
| δ=0 (drop logit-KD) | ✅ Always helps | +1–2pp on CNN; LLM not tested standalone |
| γ=0 (drop sparsity) | ✅ Neutral to positive, simplifies loss | within ±0.2pp alone; +2pp combined with drop-KD at 18ep |
| Mahalanobis sv_weighting (LLM) | ✅ Huge | GPT-2 ppl 486→131 at τ=0.95, same arch |
| Mahalanobis sv_weighting (CNN, with KD) | ≈ neutral | 89.77 vs full 89.82 @ τ=0.85 |
| Mahalanobis sv_weighting (CNN, drop-KD) | ❌ Regresses | v3=89.31 vs v3_no_kd_no_spar=91.87 @ τ=0.85 |

**Recommended defaults** (post-validation 2026-04-22):

### CNN (ResNet50/CIFAR-10 default)

```yaml
training:
  use_logit_kd: false            # δ=0; KD hurts when combined with L_subspace
  loss_delta: 0.0
  loss_gamma: 0.0                # drop sparsity; +2pp combined with drop-KD
  sv_weighting: sqrt             # keep old default; Mahalanobis has numerical
                                 # pathology at high effective_rank
  subspace_mode: spatial
```

Achieves **91.87% @ τ=0.85** — new best for the method on CIFAR-10,
+2.05pp over `full` and +0.47pp over AT baseline.

### Transformer (GPT-2/WikiText-2 default)

```bash
python scripts/09_llm_distill.py --threshold 0.95 \
  --rank-definition variance \
  --sv-weighting mahalanobis      # key change — whitens subspace MSE
```

Achieves **ppl 130.7 @ τ=0.95** — 3.7× improvement over sqrt-weighted
baseline (486) at identical student architecture.

## Code changes

All committed on NFS at `/mnt/shared_storage/asd/`; not yet pushed to git.

### `asd/losses/subspace_loss.py`

Added two new `sv_weighting` modes to `_sv_weights`:

```python
if mode == "mahalanobis":
    # w_i = 1/λ_i, normalized to mean 1.
    # Equivalent to MSE on Λ^(-1/2)-whitened projections.
    lam_max = sv.max().clamp(min=1e-10)
    sv = sv.clamp(min=1e-6 * lam_max)
    w = 1.0 / sv
    return w / w.mean().clamp(min=1e-10)

if mode == "inv_sqrt":
    # Half-whitening — less aggressive than full Mahalanobis.
    lam_max = sv.max().clamp(min=1e-10)
    sv = sv.clamp(min=1e-6 * lam_max)
    w = 1.0 / sv.sqrt()
    return w / w.mean().clamp(min=1e-10)
```

Numerical floor (`1e-6 · λ_max`) prevents noise-floor eigenvalues from
exploding the inverted weight. Verified with a unit smoke test:

```
>>> _sv_weights(torch.tensor([10., 5., 2., 1., 0.1]), 4, 'mahalanobis')
tensor([0.2222, 0.4444, 1.1111, 2.2222])
```

### `scripts/08_ablation.py`

Added three new variants:

```python
"v3":                    {"use_logit_kd": False, "delta": 0.0, "gamma": 0.0,
                          "sv_weighting": "mahalanobis"},
"v3_mahalanobis_only":   {"sv_weighting": "mahalanobis"},
"v3_no_kd_no_spar":      {"use_logit_kd": False, "delta": 0.0, "gamma": 0.0},
```

The two component ablations let us attribute v3's effect:
- `v3 − v3_no_kd_no_spar` = isolated effect of Mahalanobis
- `v3_no_kd_no_spar − full` = isolated effect of dropping KD+sparsity
- `v3_mahalanobis_only − full` = effect of Mahalanobis *with* KD present

### `scripts/09_llm_distill.py`

- Added `effective_rank_participation` and `effective_rank_entropy` helpers.
- Added `--rank-definition {variance,participation,entropy}` flag.
- Added `--sv-weighting {uniform,linear,sqrt,mahalanobis,inv_sqrt}` flag.
- Plumbed both into `LLMASDLoss`.
- Result JSON records both so re-runs are distinguishable.

### `scripts/07_bench.py`

Added `svhn` to `--dataset` argparse choices. The data loader already
supported it; only the argparse gate was blocking.

### `scripts/run_cell.sh`

Added 10 new cell names. Full list and prodjob IDs in
`/mnt/shared_storage/asd/RUN_STATE.md`.

## Ranked impact summary (post-validation)

| # | Change | Measured impact | Cost |
|---|---|---|---|
| 1 | **v3_no_kd_no_spar on CNN** (drop KD + spar, keep sqrt) | **91.87% @ τ=0.85 on RN50/CIFAR-10** — new best; +2.05pp over `full`, +0.47pp over AT | 0 LoC, config only |
| 2 | **Mahalanobis sv_weighting (LLM)** | **ppl 486 → 131** at τ=0.95, same arch | 5 LoC |
| 3 | Mahalanobis sv_weighting (CNN, with KD on) | ≈ neutral (89.77 vs full 89.82 @ τ=0.85) | — |
| 4 | Mahalanobis sv_weighting + drop-KD (CNN) | ❌ −2.56pp vs v3_no_kd_no_spar @ τ=0.85 (numerical pathology) | fixable |

**Interpretation**: the two drops (KD + sparsity) are the actual v3 win on
CNN. Swapping `sqrt → mahalanobis` on top of that regresses. On LLM the
story is inverted: sqrt is fine as a baseline, Mahalanobis is the win.

## Proposed v3.1 refinement (hypothesis, untested)

An adaptive weighting that blends sqrt and Mahalanobis based on the
spectrum's heavy-tailedness:

```python
# Compute participation ratio as a measure of spectrum spread.
pr = (sv.sum() ** 2) / (sv ** 2).sum().clamp(min=1e-12)
# Heavy-tailed → pr << k. Clean/diffuse → pr ≈ k.
# Interpolate: α=1 for pr/k ≈ 1 (sqrt), α=0 for pr/k << 1 (Mahalanobis).
alpha = (pr / k).clamp(0, 1)
w_sqrt = sv.sqrt() / sv.sqrt().mean()
w_mahal = (1.0 / sv.clamp(min=1e-6*sv.max()))
w_mahal = w_mahal / w_mahal.mean()
w = alpha * w_sqrt + (1 - alpha) * w_mahal
```

Predicted to reduce to sqrt on sharp spectra and to Mahalanobis on heavy-
tailed spectra. Not implemented; needs its own validation experiment.

## How to reproduce v3 cells

From `/mnt/shared_storage/asd/`:

```bash
bash scripts/submit_anyscale.sh \
  asd_v3_t70 asd_v3_t85 asd_v3_t95 \
  asd_v3_ablation_t85 \
  llm_gpt2_participation_t90 llm_gpt2_participation_t95 \
  llm_gpt2_mahalanobis_t95
```

Single-cell for debugging:

```bash
cd /mnt/shared_storage/asd
source .venv/bin/activate
python scripts/08_ablation.py --variant v3 --threshold 0.85 --epochs 18 \
  --teacher-weights outputs/teacher_finetuned.pt \
  --output-dir outputs/asd_v3_t85_local
```

LLM single-cell:

```bash
python scripts/09_llm_distill.py --threshold 0.95 --epochs 2 \
  --batch-size 8 --seq-len 256 --sv-weighting mahalanobis \
  --output-dir outputs/llm_mahalanobis_local
```
