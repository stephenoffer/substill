# F-ASD quickstart

Three steps: profile, build, train. F-ASD works on decoder-only LLM
teachers out of the box (GPT-2 and Llama-family; register a custom
detector for others). For the full mechanism and design rationale see
[algorithm.md](algorithm.md).

## 1. Profile

```python
import fasd

profile = fasd.profile(
    teacher,
    calibration_loader,
    mode="branch",           # branchwise profiling (attn.q, attn.k, attn.v, attn.o, ffn.up, ffn.gate, ffn.down)
    rank_tol=0.02,           # max KL between unpatched / patched teacher for the behavioral rank search
    token_weighting="entropy",
)
profile.save("teacher.fasd")         # pickle-safe round-trip
# ...
profile = fasd.TeacherProfile.load("teacher.fasd")
```

`fasd.profile` runs the teacher over the calibration loader,
accumulates per-branch covariances, eigendecomposes them, and then
calls `choose_behavioral_rank` on each branch to pick the smallest
rank whose projection preserves teacher logits within `rank_tol`.

The returned `TeacherProfile.branches` is a list of `BranchProfile`
objects with behavioral ranks, variance ranks (for comparison),
eigenvectors, eigenvalues, and the full KL curve for each tested
rank.

## 2. Build

```python
student = fasd.build_student(
    teacher,
    profile,
    absorbed_init=True,      # W_s = V_out^T W_T V_in
    template="auto",         # auto-detects gpt2 or llama
    arch_multiplier=1.0,     # purely profile-driven sizing
)
```

`build_student` derives a compressed transformer config from the
profile (width-first, retains attention heads, contiguous depth
drops) and fills student linear weights by absorbing teacher weights
through the retained bases. For GPT-2 this covers `c_attn`,
`c_proj`, `c_fc`, `c_proj`, embeddings, position table, and layer
norms. For Llama it covers `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`, embeddings, RMSNorms, and the
LM head.

## 3. Train

### Option A — attach `F_ASDLoss` to your own loop

```python
loss_fn = fasd.F_ASDLoss(
    profile,
    objective="procrustes",  # or "gram" / "cka"
    schedule=fasd.default_schedule(),
).to(device)

optimizer = torch.optim.AdamW(
    list(student.parameters()) + list(loss_fn.parameters()),
    lr=5e-5,
)

for step, batch in enumerate(train_loader):
    with fasd.capture(teacher, profile, detach=True) as t_hid:
        with torch.no_grad():
            teacher(**batch)
    with fasd.capture(student, profile) as s_hid:
        s_out = student(**batch)
    s_logits, t_logits = s_out.logits, ...

    sub_loss = loss_fn(
        dict(s_hid.items()), dict(t_hid.items()),
        step_frac=step / total_steps,
    )
    kd_loss = fasd.skew_kl(s_logits[:, :-1], t_logits[:, :-1])
    task_loss = ...
    total = task_loss + 0.5 * sub_loss + kd_loss
    total.backward()
    optimizer.step()
    optimizer.zero_grad()

    # End of warm-up — fold the learned projectors away.
    if step == total_steps // 10:
        loss_fn.fold_projectors_into_(student)
```

### Option B — one-call multi-stage driver

```python
result = fasd.distill(
    teacher,
    student,
    train_loader,
    profile=profile,
    generative_kd="skew_kl",
    on_policy_start=0.5,          # on-policy from 50% of training
    on_policy_ratio=0.5,
    teacher_correction_steps=200, # short teacher adaptation before profiling
    quantize=True,                # AWQ + QAD final stage
    total_steps=1000,
    rollout_prompts=prompts,      # prompts for student.generate during on-policy stage
)
print(result.best_metric, result.teacher_metric)
```

## Scaling notes

For teachers with `hidden_size >= 1024` or `> 1B` parameters,
`fasd.profile` auto-selects the randomized-SVD backend of
`StreamingPCA`. To use feature caching (KDFlow-style) and avoid
rerunning the teacher for every feature-loss batch, pass
`cache_teacher_features=True` to `fasd.distill`.

## Common pitfalls

- **Loss diverges on LLMs**: use `objective="procrustes"` (or start
  with Gram/CKA via the default schedule) and keep
  `normalize_features=True`.
- **`capture()` returns empty hiddens**: the student must have the
  same module paths as the teacher, or pass
  `branches=[BranchSpec(...)]` explicitly.
- **`autodetect_branches` raises `NotImplementedError`**: pass
  branches explicitly or register a detector via
  `fasd.register_detector("my_model", my_detector)`.
