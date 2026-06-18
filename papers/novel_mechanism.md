# CPSD: Circuit-Preserving Subspace Distillation — formal mechanism

**Date:** 2026-06-17
**Purpose:** Phase 0.2 deliverable. Formal statement of the novel method, the unclaimed
conjunction, and proof sketches for the two load-bearing claims (exact-at-init and the
circuit-distortion bound). Companion to `gap_analysis.md`.

Notation: teacher hidden dim `d` (= `d_T`), student hidden dim `r` (= `d_S`), per-head dim
`d_h`, heads `H`, KV groups `G`, heads-per-group `m = H/G`. Teacher weights are frozen and
written `W_T`. Orthonormal bases (columns) are `V` with `Vᵀ V = I`. Activations on the
residual stream are `x ∈ ℝ^d`; the compressed student carries `x_S ∈ ℝ^r`.

---

## 1. Component CPI — circuit-preserving initialization

### 1.1 The failure mode it fixes

Per-matrix absorbed init compresses each projection independently:
`W_Q ↦ V_qᵀ W_Q V_r`, `W_K ↦ V_kᵀ W_K V_r`, with **different** output bases `V_q ≠ V_k`.
The teacher attention score between query head `h` and key group `g` is
`s = qₕᵀ kg = (W_Q x_i)ₕᵀ (W_K x_j)g`. In the student, `q_S = V_qᵀ qₕ`, `k_S = V_kᵀ kg`, so
the student score is

```
s_S = q_Sᵀ k_S = qₕᵀ V_q V_kᵀ kg .
```

This equals the teacher score `qₕᵀ kg` only if `V_q V_kᵀ = I` on the relevant directions —
i.e. only if `V_q = V_k`. With independent bases it does not, which is the documented
3–7% GQA PPL regression and the general SVD-LLM/ASVD circuit-breaking failure.

### 1.2 The shared-subspace construction (what we do; NOT operator-SVD)

For each layer ℓ and KV group g, collect post-projection (pre-RoPE, see §1.4) activations
for all `m` query heads in the group plus its key and value:

```
X_{ℓ,g} = stack(q_{h_1}, …, q_{h_m}, k_g, v_g) ∈ ℝ^{(m+2)·N × d_h},
```

and take the eigenvectors of `cov(X_{ℓ,g}) = X_{ℓ,g}ᵀ X_{ℓ,g}` as a **single** orthonormal
basis `V_{ℓ,g} ∈ ℝ^{d_h × d_h}` (this is exactly `gqa_basis.collect_gqa_bases`). Using the
top-`d'_h` columns of `V_{ℓ,g}` as the output basis for **every** query head in group g and
for k_g, v_g makes `V_q = V_k = V_v = V_{ℓ,g}`, so the cancellation in §1.1 holds.

**This is the shared-subspace method, not KQ-SVD.** We do *not* form the operator `W_Qᵀ W_K`
and compute its best rank-r SVD (that is KQ-SVD's claim, with its own bound). We preserve the
circuit by *sharing the activation subspace*, which is a different and already-implemented
primitive. Keeping this distinction explicit is what protects the novelty claim.

### 1.3 The OV circuit (the unclaimed weight-side delta)

The value circuit is `W_O (A W_V x)` where `A` is the attention matrix. Per head, the
composition `W_O^h W_V^h` is the OV circuit operator. Absorbing `W_V` with output basis
`V_o` and `W_O` with *input* basis `V_o` (the same basis on the value/attention-output
space) preserves the OV composition by the identical `V_oᵀ V_o = I` cancellation:

```
o_S = (V_oᵀ W_O^h)(A (V_oᵀ W_V^h x))   vs   o = W_O^h (A W_V^h x),
```

and `W_O^h V_o V_oᵀ W_V^h → W_O^h W_V^h` on the retained subspace. KQ-SVD does **not** treat
the OV circuit, and it operates on the KV cache, not the weights — so the OV + weight-side
construction is the surviving CPI delta, used as the initialization for §2.

### 1.4 RoPE-awareness (correctness prerequisite)

The §1.2 cancellation uses `V_q = V_k`, but with RoPE the score is
`s = (R_i qₕ)ᵀ (R_j kg) = qₕᵀ R_iᵀ R_j kg = qₕᵀ R(θ, j−i) kg`. After the basis change the
student computes `(R_i V qₕ_S)ᵀ (R_j V kg_S) = q_Sᵀ Vᵀ R(θ,j−i) V k_S`, which equals the
teacher score **only if `Vᵀ R(θ,Δ) V = R(θ,Δ)` for all Δ**, i.e. `V` commutes with every
RoPE rotation. A general PCA basis does not. The existing claim in
`gqa_basis.py:33-37` that "RoPE commutes with the basis" is therefore **false**, and the
diagnostic only tests Δ-free (no-RoPE) scores.

**Fix (Phase 0.3c / 2.1).** RoPE acts as a block-diagonal rotation on the `d_h/2` 2D
coordinate planes. We restrict `V` to commute with this structure by one of:
(i) **decoupled** — split each head into RoPE dims and non-RoPE/NoPE dims (à la Palu RoRoPE),
apply the shared subspace only to the part RoPE acts trivially on, keep RoPE dims uncompressed
or compressed in a rotation-commuting way; or (ii) **plane-block-diagonal `V`** — constrain
`V` to be block-diagonal over RoPE's 2D planes so it commutes with `R(θ,Δ)` by construction.
The acceptance test is the *post-RoPE* score residual (extend `attention_score_residual` to
apply RoPE before scoring), which the current code omits.

---

## 2. Component MT — manifold-constrained end-to-end training

### 2.1 Why FactoredLinear does not drop in

`FactoredLinear.forward` computes `y = ((x U_in) Bᵀ) U_outᵀ` with `U_in ∈ ℝ^{d×k}`, i.e. it
consumes **teacher-dim** input `x ∈ ℝ^d`. The compressed student's residual stream is
already `x_S ∈ ℝ^r`, so a naive replacement is dimensionally inconsistent. This is the
deferral noted in `factored_linear.py:7-9`.

### 2.2 Correct construction (frozen teacher, trainable Stiefel factors)

Hold the teacher weight `W_T` **frozen** and expose the bases as trainable Stiefel
parameters `V_in ∈ St(d, r)`, `V_out ∈ St(d, r_out)`. The student edge computes

```
y_S = V_outᵀ W_T V_in x_S          (materialize-each-step form), or equivalently
y_S = V_outᵀ ( W_T ( V_in x_S ) )  (route-through-teacher-dim form, no materialization).
```

- **Inference compression is preserved:** after training, fold the factors back to
  `W_S = V_outᵀ W_T V_in ∈ ℝ^{r_out × r}` and deploy a plain compressed linear. The student
  runs in `r`-dim space at inference.
- **The factors train end-to-end against the KD loss:** gradients flow to `V_in, V_out`,
  which `StiefelAdam` keeps on the manifold via Cayley retraction (existing
  `stiefel_optim.py`, `stiefel_param_groups`). This is the Pillar-2 trainability that is
  currently provided only by the RR-Norm `Q`; here the QK/OV bases themselves adapt.
- **Cost / risk:** the route-through-teacher-dim form costs `O(d·r + d·d_out + d_out·r)`
  per token vs `O(r·r_out)` for a collapsed linear — extra train-time compute and the
  activation memory of teacher-dim intermediates. This is precisely why it was deferred and
  is the central tractability question (Phase 0.3b). At inference there is no overhead.

A `TeacherFactoredLinear` module (frozen `W_T`, trainable Stiefel `V_in/V_out`, optional
block-diagonal correction `S`) is the clean home for this; it reuses `FactoredLinear`'s
Stiefel markers and `effective_weight()` folding.

---

## 3. Component DDR — distillation-driven differentiable rank

Replace the frozen per-edge rank (behavioral-rank threshold; Fisher knapsack) with a soft,
differentiable column gate. For edge e with ordered basis columns, define gate
`σ(α_{e,i}) ∈ (0,1)` on column i with learnable logits `α_e`, so the effective basis is
`Ṽ_e = V_e diag(σ(α_e))`. The training objective is

```
min  L_KD(student(α, V))  +  λ · ( Σ_e Σ_i σ(α_{e,i}) · cost_{e,i}  −  P* )_+
```

where `cost_{e,i}` is the per-column parameter cost (exact, from edge dims; reuses
`rank_allocator` cost logic) and `P*` is the global budget. The gate is optimized **against
the KD loss** (not reconstruction — the Dobi-SVD/LLRC delta) and is shared across the
circuit-preserving factors of §1 and across **per-expert** edges for MoE (each expert edge
is already a separate `EdgeSpec` from `edges_from_profile`). At the end of training the gates
are hardened (threshold σ) to integer ranks and the factors folded (§2.2).

Stability risk (Phase 0.3a): backprop through a soft gate composed with the SVD-derived
basis can be ill-conditioned; Dobi-SVD needed a Taylor-expansion stabilization. We test
whether a straight-through or temperature-annealed gate is stable in our setting before
committing.

---

## 4. Proof sketches

### 4.1 Exact at init (full rank)

**Claim.** If `V_in, V_out` are complete orthonormal bases (`r = d`, `r_out = d_out`, so
`V Vᵀ = I`), then the absorbed/factored student reproduces the teacher edge exactly:
`W_S = V_outᵀ W_T V_in` gives effective weight `V_out W_S V_inᵀ = V_out V_outᵀ W_T V_in V_inᵀ
= W_T`. Composed across edges with matching shared bases at each boundary, the student logits
equal the teacher logits. *(This is the existing `test_fasd_absorbed_init.py` invariant; CPI
preserves it because shared bases are still orthonormal.)*

**Reduced rank.** With `r < d`, `V Vᵀ = P` is the orthogonal projector onto the retained
subspace, and the effective weight is `P_out W_T P_in` — the projection of `W_T` onto the
activation principal subspaces. This is **not** the best low-rank approximation of `W_T`
(that would be its SVD); it is the best preserver of the *activations*, which is the correct
objective for distillation. The init error is `‖W_T − P_out W_T P_in‖`, bounded by the tail
energy `Σ_{i>r} λ_i` of the activation covariance — near-zero when retained directions
capture the activation energy. So "exact at init" is precise only at full rank; at reduced
rank it is the controlled activation-subspace projection.

### 4.2 Circuit-distortion bound (QK and OV)

**Claim (shared subspace).** Let `V` be the shared per-group basis with projector `P = V Vᵀ`
(rank `d'_h`). The per-group attention-score distortion is bounded by the activation tail
energy:

```
‖S_teacher − S_student‖_F  ≤  C · ( Σ_{i > d'_h} λ_i^{(g)} ) · ‖q‖ ‖k‖ / (energy)
```

where `λ_i^{(g)}` are the eigenvalues of the joint group covariance `cov(X_{ℓ,g})` and `C` is
an O(1) constant. Sketch: `s_student = qᵀ P_q P_k k`; with `P_q = P_k = P` (shared basis),
`s_teacher − s_student = qᵀ(I − P)... k`-type terms, each factor bounded in operator norm by
the projector complement onto the discarded directions, whose energy is the discarded
eigenvalue mass. The same argument with `V_o` gives the OV-circuit bound for
`W_O^h (I − P) W_V^h`. *(This is a tail-energy bound, weaker than KQ-SVD's operator-optimal
bound, but it applies to the **shared-subspace** construction and extends to the OV circuit;
the discriminating test in Phase 2.4 verifies it empirically on a synthetic model with known
circuit structure.)*

**RoPE caveat.** The bound above is for the pre-RoPE / RoPE-commuting case. With a general
`V` and RoPE, an additional term `qᵀ Vᵀ[R(θ,Δ), V]... k` (the commutator) appears and is
**not** controlled by tail energy — which is exactly why §1.4's RoPE-aware `V` is required
for the bound to hold on RoPE models.

---

## 5. The conjunction (restate for ablations)

CPSD = CPI (§1) ∘ MT (§2) ∘ DDR (§3). The paper's central experiment must show the
*conjunction* beats:
- **Dobi-SVD** (isolates: KD-driven rank + Stiefel factors vs reconstruction-driven rank),
- **KQ-SVD** (isolates: OV + weight-side + trained factors vs QK-only frozen KV-cache SVD),
- **DistiLLM-2 / Minitron** (isolates: circuit-preserving manifold-trained factors vs KD on
  a fixed / separately-pruned architecture),
- **RFID-MoE** (isolates: CPSD-in-experts vs routing×info-density allocation alone).

Ablation cells: CPI-only, CPI+MT, CPI+DDR, full CPSD — to attribute gain to the conjunction,
not any single component (each of which is individually anticipated, per `gap_analysis.md`).
