#!/usr/bin/env python3
"""F-ASD ablation entrypoint used by the Anyscale job matrix.

Reads the rung name from ``FASD_RUNG`` env var and runs the matching
configuration on a GPT-2 teacher + WikiText-2 corpus. Writes a
single-line JSON result to stdout for easy aggregation.

v11 ladder — Periodic Re-Absorption (PRA) with matched compression.
Each rung runs at the SAME parameter count (set by ``--target-compression``),
so cross-rung comparison is apples-to-apples instead of confounded by
rank-allocator slack as in v10.

Rungs:

    r0_random            — random-init student, KD only (compression floor)
    r1_static            — absorbed init + procrustes schedule + skew KL (v10 winner)
    r2_pra200            — r1 + Periodic Re-Absorption every 200 steps (NOVELTY)
    r3_pra100            — r1 + PRA every 100 steps (frequency sensitivity)
    r4_pra200_onpolicy   — r2 + on-policy rollouts (test PRA rescues drift)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fasd  # noqa: E402


RUNG_ORDER = [
    "r0_random",
    "r1_static",
    "r2_pra200",
    "r3_pra100",
    "r4_pra200_onpolicy",
    # v10 rungs retained so old TAGs still render in progress.md.
    "0_baseline",
    "1_behavioral",
    "2_procrustes",
    "3_skewkl",
    "4_absorbed",
    "5_onpolicy",
    "6_quantize",
    "7_full",
]


def _resolve_output_dir(tag: str) -> str:
    explicit = os.environ.get("FASD_OUTPUT_DIR")
    if explicit:
        return explicit
    # Anyscale exports ANYSCALE_ARTIFACT_STORAGE to a persistent S3-backed path
    # that survives cluster teardown — prefer it so `so we can pick it up` works
    # across cluster restarts.
    base = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
    if base:
        return os.path.join(base, "fasd", tag)
    return os.path.join(os.getcwd(), "outputs", "fasd", tag)


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _atomic_write(path: str, data: str) -> None:
    # S3: write to a local temp then `aws s3 cp` — Python's open() can't speak s3://,
    # which is why v10 results were lost.
    if _is_s3(path):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp") as f:
            f.write(data)
            local = f.name
        try:
            subprocess.run(
                ["aws", "s3", "cp", "--quiet", local, path],
                check=True,
            )
        finally:
            try:
                os.unlink(local)
            except FileNotFoundError:
                pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Per-pid tmp name so concurrent writers on shared storage don't collide.
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def _path_exists(path: str) -> bool:
    if _is_s3(path):
        # `aws s3 ls <full-key>` returns 0 with output if exact key exists,
        # 1 (or 0 with no output) otherwise.
        r = subprocess.run(
            ["aws", "s3", "ls", path],
            capture_output=True, text=True,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    return os.path.exists(path)


def _list_dir(path: str) -> list[str]:
    """List filenames in ``path`` (S3 or local). Returns empty if missing."""
    if _is_s3(path):
        d = path.rstrip("/") + "/"
        r = subprocess.run(["aws", "s3", "ls", d], capture_output=True, text=True)
        if r.returncode != 0:
            return []
        names = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts and not parts[-1].endswith("/"):
                names.append(parts[-1])
        return names
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def _read_text(path: str) -> str:
    if _is_s3(path):
        r = subprocess.run(
            ["aws", "s3", "cp", "--quiet", path, "-"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout
    with open(path) as f:
        return f.read()


def _remove_if_exists_any(path: str) -> None:
    if _is_s3(path):
        # rm is idempotent for S3 (no error if missing).
        subprocess.run(["aws", "s3", "rm", "--quiet", path], check=False)
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _write_json(path: str, data: dict) -> None:
    _atomic_write(path, json.dumps(data, indent=2, sort_keys=True, default=str))


def _fmt(v, spec: str = ".2f", default: str = "—") -> str:
    if v is None:
        return default
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return default


def _render_progress_md(output_dir: str, tag: str) -> str:
    results_dir = os.path.join(output_dir, "results")
    status_dir = os.path.join(output_dir, "status")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Discover all variant keys present in the results / status dirs. A "key"
    # is a base rung name, optionally suffixed with -c<compression>. We render
    # one row per variant rather than one row per RUNG_ORDER entry, so v11
    # matched-compression sweeps surface every (rung × compression) cell.
    variants: set[str] = set()
    for fn in _list_dir(results_dir):
        if fn.endswith(".json"):
            variants.add(fn[:-5])
    for fn in _list_dir(status_dir):
        if fn.endswith(".running.json"):
            variants.add(fn[: -len(".running.json")])
        elif fn.endswith(".failed.json"):
            variants.add(fn[: -len(".failed.json")])
    # Always include canonical rungs (so a freshly-empty tag still shows pending).
    for r in RUNG_ORDER:
        variants.add(r)

    def _sort_key(v: str):
        # Group by base rung order, then by compression suffix (numerically).
        base, _, suffix = v.partition("-c")
        try:
            order_idx = RUNG_ORDER.index(base)
        except ValueError:
            order_idx = len(RUNG_ORDER)
        try:
            comp = float(suffix) if suffix else 0.0
        except ValueError:
            comp = -1.0
        return (order_idx, comp, v)

    rows = []
    missing = []
    for rung in sorted(variants, key=_sort_key):
        result_path = os.path.join(results_dir, f"{rung}.json")
        failed_path = os.path.join(status_dir, f"{rung}.failed.json")
        running_path = os.path.join(status_dir, f"{rung}.running.json")
        if _path_exists(result_path):
            r = json.loads(_read_text(result_path))
            rows.append(
                f"| {rung} | done | "
                f"{_fmt(r.get('final_student_ppl'))} | "
                f"{_fmt(r.get('teacher_ppl'))} | "
                f"{_fmt(r.get('compression_ratio'))}x | "
                f"{_fmt(r.get('train_time_s'), '.1f')}s | — |"
            )
        elif _path_exists(failed_path):
            r = json.loads(_read_text(failed_path))
            err_lines = (r.get("error") or "").strip().splitlines()
            err_msg = err_lines[-1][:80] if err_lines else "unknown"
            rows.append(f"| {rung} | failed | — | — | — | — | {err_msg} |")
            missing.append(rung)
        elif _path_exists(running_path):
            r = json.loads(_read_text(running_path))
            rows.append(
                f"| {rung} | running | — | — | — | — | "
                f"started {r.get('started_at', '?')} |"
            )
        else:
            rows.append(f"| {rung} | pending | — | — | — | — | — |")
            missing.append(rung)

    lines = [
        f"# F-ASD Ablation Progress — tag `{tag}`",
        "",
        f"Updated: {now}",
        "",
        "| Rung | Status | Final PPL | Teacher PPL | Compression | Train time | Notes |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]
    if missing:
        lines += [
            "",
            "## Resume missing rungs",
            "",
            "```bash",
            f'TAG={tag} RUNGS="{" ".join(missing)}" bash scripts/fasd_ablation_submit.sh',
            "```",
        ]
    return "\n".join(lines) + "\n"


def _update_progress_md(output_dir: str, tag: str) -> None:
    path = os.path.join(output_dir, "progress.md")
    _atomic_write(path, _render_progress_md(output_dir, tag))


# Each rung has the same compression target (set by --target-compression on
# the CLI, applied to all rungs). The rung config knobs below are purely
# *algorithmic* — what the student does given a fixed param budget.
RUNG_CONFIGS = {
    "r0_random": dict(
        use_behavioral=True,
        objective="gram",
        generative_kd="forward_kl",
        absorbed_init=False,
        on_policy_start=2.0,
        quantize=False,
        reabsorb_every_steps=0,
    ),
    "r1_static": dict(
        use_behavioral=True,
        objective="schedule",
        generative_kd="skew_kl",
        absorbed_init=True,
        on_policy_start=2.0,
        quantize=False,
        reabsorb_every_steps=0,
    ),
    "r2_pra200": dict(
        use_behavioral=True,
        objective="schedule",
        generative_kd="skew_kl",
        absorbed_init=True,
        on_policy_start=2.0,
        quantize=False,
        reabsorb_every_steps=200,
    ),
    "r3_pra100": dict(
        use_behavioral=True,
        objective="schedule",
        generative_kd="skew_kl",
        absorbed_init=True,
        on_policy_start=2.0,
        quantize=False,
        reabsorb_every_steps=100,
    ),
    "r4_pra200_onpolicy": dict(
        use_behavioral=True,
        objective="schedule",
        generative_kd="skew_kl",
        absorbed_init=True,
        on_policy_start=0.5,
        quantize=False,
        reabsorb_every_steps=200,
    ),
    # v10 rungs retained for backward-compat re-runs of old tags.
    "0_baseline": dict(
        use_behavioral=False, objective="gram", generative_kd="forward_kl",
        absorbed_init=False, on_policy_start=2.0, quantize=False, reabsorb_every_steps=0,
    ),
    "1_behavioral": dict(
        use_behavioral=True, objective="gram", generative_kd="forward_kl",
        absorbed_init=False, on_policy_start=2.0, quantize=False, reabsorb_every_steps=0,
    ),
    "2_procrustes": dict(
        use_behavioral=True, objective="schedule", generative_kd="forward_kl",
        absorbed_init=False, on_policy_start=2.0, quantize=False, reabsorb_every_steps=0,
    ),
    "3_skewkl": dict(
        use_behavioral=True, objective="schedule", generative_kd="skew_kl",
        absorbed_init=False, on_policy_start=2.0, quantize=False, reabsorb_every_steps=0,
    ),
    "4_absorbed": dict(
        use_behavioral=True, objective="schedule", generative_kd="skew_kl",
        absorbed_init=True, on_policy_start=2.0, quantize=False, reabsorb_every_steps=0,
    ),
    "5_onpolicy": dict(
        use_behavioral=True, objective="schedule", generative_kd="skew_kl",
        absorbed_init=True, on_policy_start=0.5, quantize=False, reabsorb_every_steps=0,
    ),
    "6_quantize": dict(
        use_behavioral=True, objective="schedule", generative_kd="skew_kl",
        absorbed_init=True, on_policy_start=2.0, quantize=True, reabsorb_every_steps=0,
    ),
    "7_full": dict(
        use_behavioral=True, objective="schedule", generative_kd="skew_kl",
        absorbed_init=True, on_policy_start=0.5, quantize=True, reabsorb_every_steps=0,
    ),
}


def get_dataloaders(batch_size: int, seq_len: int):
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    class _WT2(Dataset):
        def __init__(self, split, tokenizer, seq_len):
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [t for t in ds["text"] if t.strip()]
            ids = tokenizer.encode("\n\n".join(texts))
            n = len(ids) // seq_len
            self.tokens = torch.tensor(
                ids[: n * seq_len], dtype=torch.long
            ).view(n, seq_len)

        def __len__(self):
            return self.tokens.shape[0]

        def __getitem__(self, idx):
            t = self.tokens[idx]
            return {
                "input_ids": t,
                "labels": t,
                "attention_mask": torch.ones_like(t),
            }

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    return {
        "train": DataLoader(_WT2("train", tok, seq_len), batch_size=batch_size, shuffle=True),
        "val": DataLoader(_WT2("validation", tok, seq_len), batch_size=batch_size),
    }


def _eval_ppl(model, loader, device):
    import torch.nn.functional as F

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total += float(loss.item()) * shift_labels.numel()
            n += shift_labels.numel()
    return float(torch.exp(torch.tensor(total / n)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", default=os.environ.get("FASD_RUNG", "0_baseline"))
    parser.add_argument("--tag", default=os.environ.get("FASD_TAG", "local"))
    parser.add_argument("--teacher", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank-tol", type=float, default=0.02)
    parser.add_argument("--max-rank", type=int, default=512)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument(
        "--arch-multiplier", type=float,
        default=float(os.environ.get("FASD_ARCH_MULT", 1.0)),
        help="Scale all behavioral ranks by this factor before student-config "
             "derivation. <1.0 compresses the residual stream and FFN intermediate. "
             "Ignored if --target-compression is provided.",
    )
    parser.add_argument(
        "--target-compression", type=float,
        default=float(os.environ.get("FASD_TARGET_COMPRESSION", 0.0)) or None,
        help="If set (e.g. 2.0 or 4.0), binary-search arch_multiplier to hit "
             "this teacher/student parameter ratio. Replaces --arch-multiplier.",
    )
    parser.add_argument(
        "--reabsorb-every-steps", type=int,
        default=int(os.environ.get("FASD_REABSORB_EVERY", 0)) or 0,
        help="If >0, refresh teacher PCA and re-project student weights every N "
             "steps (Periodic Re-Absorption). 0 disables. Per-rung default in "
             "RUNG_CONFIGS may override this when the CLI value is 0.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = RUNG_CONFIGS[args.rung]
    print(f"[fasd-ablation] rung={args.rung} tag={args.tag} config={config}")

    output_dir = _resolve_output_dir(args.tag)
    # Embed compression target in the result filename so a sweep across
    # multiple targets doesn't have rungs clobber each other.
    rung_key = args.rung
    if args.target_compression and args.target_compression > 0:
        rung_key = f"{args.rung}-c{args.target_compression:.1f}"
    result_path = os.path.join(output_dir, "results", f"{rung_key}.json")
    running_path = os.path.join(output_dir, "status", f"{rung_key}.running.json")
    failed_path = os.path.join(output_dir, "status", f"{rung_key}.failed.json")
    print(f"[fasd-ablation] output_dir={output_dir}")

    if _path_exists(result_path):
        print(f"[fasd-ablation] skipping — result already at {result_path}")
        _update_progress_md(output_dir, args.tag)
        return

    _write_json(
        running_path,
        {
            "rung": args.rung,
            "tag": args.tag,
            "config": config,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": os.environ.get("HOSTNAME", ""),
            "anyscale_job": os.environ.get("ANYSCALE_JOB_NAME", ""),
        },
    )
    _update_progress_md(output_dir, args.tag)

    try:
        _run_rung(args, config, result_path)
    except Exception:
        _write_json(
            failed_path,
            {
                "rung": args.rung,
                "tag": args.tag,
                "config": config,
                "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error": traceback.format_exc(),
            },
        )
        _remove_if_exists_any(running_path)
        _update_progress_md(output_dir, args.tag)
        raise

    _remove_if_exists_any(running_path)
    _remove_if_exists_any(failed_path)
    _update_progress_md(output_dir, args.tag)


def _find_arch_multiplier(teacher, profile, target_compression: float) -> float:
    """Binary-search arch_multiplier to hit a target teacher/student param ratio.

    arch_multiplier scales behavioral_ranks up toward teacher hidden size.
    Upper bound is dynamic: t_hidden / min(behavioral_rank) (clamped to 32) so
    we can always reach a full-rank student even when ranks are small.
    Returns the multiplier whose student is closest to ``target_params``.
    """
    teacher_params = sum(p.numel() for p in teacher.parameters())
    target_params = teacher_params / max(1.0, float(target_compression))
    t_hidden = int(getattr(teacher.config, "n_embd", getattr(teacher.config, "hidden_size", 0)))
    min_rank = max(
        1,
        min(
            (int(b.behavioral_rank) for b in profile.branches if b.behavioral_rank > 0),
            default=1,
        ),
    )
    upper = max(2.0, min(32.0, float(t_hidden) / float(min_rank)))
    lo, hi = 0.05, float(upper)
    best_mult, best_err, best_n = 1.0, float("inf"), 0
    for _ in range(20):
        mid = (lo + hi) / 2
        trial = fasd.build_student(
            teacher, profile, absorbed_init=False, template="gpt2",
            arch_multiplier=mid,
        )
        n = sum(p.numel() for p in trial.parameters())
        err = abs(n - target_params)
        if err < best_err:
            best_mult, best_err, best_n = mid, err, n
        if n < target_params:
            lo = mid
        else:
            hi = mid
        del trial
    achieved = teacher_params / max(1, best_n)
    print(
        f"[fasd-ablation] matched-compression search: target={target_compression:.2f}x "
        f"target_params={target_params/1e6:.1f}M  chosen mult={best_mult:.4f}  "
        f"n={best_n/1e6:.1f}M  actual={achieved:.2f}x  "
        f"(search range=[0.05,{upper:.2f}])",
        flush=True,
    )
    return float(best_mult)


def _stage(label: str, t0_holder: list) -> None:
    """Print a stage marker with elapsed seconds since the previous marker."""
    now = time.time()
    elapsed = now - t0_holder[0] if t0_holder else 0.0
    print(f"[fasd-stage] {label} (+{elapsed:.1f}s)", flush=True)
    t0_holder[:] = [now]


def _run_rung(args, config, result_path: str) -> None:
    from transformers import GPT2LMHeadModel

    stage_t = [time.time()]
    _stage("load_teacher", stage_t)
    teacher = GPT2LMHeadModel.from_pretrained(args.teacher).to(args.device).eval()

    _stage("get_dataloaders", stage_t)
    loaders = get_dataloaders(args.batch_size, args.seq_len)

    _stage("collect_calib", stage_t)
    calib = []
    for i, b in enumerate(loaders["train"]):
        if i >= args.calib_batches:
            break
        calib.append({k: v.to(args.device) for k, v in b.items()})

    # Profile: behavioral branchwise + per-block residual.
    # The residual branches give absorbed_init a true residual-stream PCA for V_r;
    # v8 used a Frankenstein average of attn.o + ffn.down PCAs which is not
    # mathematically the residual subspace.
    from fasd.autodetect import autodetect_branches

    branchwise = list(autodetect_branches(teacher, mode="branch"))
    residual = list(autodetect_branches(teacher, mode="residual"))
    combined_branches = branchwise + residual
    _stage(f"profile_start n_branches={len(combined_branches)}", stage_t)

    t0 = time.time()
    profile = fasd.profile(
        teacher,
        calib,
        branches=combined_branches,
        rank_tol=args.rank_tol if config["use_behavioral"] else 1.0,
        token_weighting="entropy",
        max_rank=args.max_rank,
        n_calib_batches=len(calib),
        behavioral_calib_batches=min(4, len(calib)),
        search="bisect",
        device=args.device,
    )
    profile_time = time.time() - t0
    _stage(f"profile_done in {profile_time:.1f}s", stage_t)

    # When use_behavioral=False we just override behavioral_rank with variance_rank
    # to emulate the baseline without re-implementing the whole pipeline.
    if not config["use_behavioral"]:
        import dataclasses

        new_branches = [
            dataclasses.replace(b, behavioral_rank=int(b.variance_rank))
            for b in profile.branches
        ]
        profile = fasd.TeacherProfile(branches=new_branches, meta=profile.meta)

    # Resolve arch_multiplier: matched-compression search overrides the CLI knob.
    if args.target_compression and args.target_compression > 0:
        _stage("arch_multiplier_search", stage_t)
        arch_mult = _find_arch_multiplier(teacher, profile, args.target_compression)
    else:
        arch_mult = float(args.arch_multiplier)

    # Build student.
    _stage(f"build_student arch_mult={arch_mult:.4f} absorbed={config['absorbed_init']}", stage_t)
    t0 = time.time()
    student = fasd.build_student(
        teacher, profile, absorbed_init=config["absorbed_init"], template="gpt2",
        arch_multiplier=arch_mult,
    ).to(args.device)
    build_time = time.time() - t0
    _stage(f"build_student_done in {build_time:.1f}s", stage_t)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"[fasd-ablation] teacher_params={teacher_params/1e6:.1f}M  student_params={student_params/1e6:.1f}M  cmp={teacher_params/student_params:.2f}x", flush=True)

    _stage("eval_initial_student_ppl", stage_t)
    initial_student_ppl = _eval_ppl(student, loaders["val"], args.device)
    _stage("eval_teacher_ppl", stage_t)
    teacher_ppl = _eval_ppl(teacher, loaders["val"], args.device)
    _stage("eval_done", stage_t)

    # Map the rung's objective knob onto the driver's (schedule, loss_objective).
    # "schedule" → use default_schedule (gram→cka→procrustes).
    # "gram"|"cka"|"procrustes" → no schedule, single objective throughout.
    schedule = None
    use_default = False
    if config["objective"] == "schedule":
        schedule = fasd.default_schedule()
        loss_objective = "procrustes"
        use_default = True
    else:
        loss_objective = config["objective"]
        use_default = False

    # Rollout prompts for on-policy stage.
    rollout_prompts = None
    if config["on_policy_start"] < 1.0:
        first_batch = next(iter(loaders["train"]))
        rollout_prompts = first_batch["input_ids"][:, :16].to(args.device)

    t0 = time.time()
    loss_fn = fasd.F_ASDLoss(
        profile,
        objective=loss_objective,
        schedule=schedule,
    ).to(args.device)

    # Resolve PRA frequency: CLI override (>0) wins over rung default.
    pra_steps = int(args.reabsorb_every_steps) or int(config.get("reabsorb_every_steps") or 0)
    _stage(f"distill_start pra_steps={pra_steps} total_steps={args.total_steps}", stage_t)

    result = fasd.distill(
        teacher,
        student,
        loaders["train"],
        profile=profile,
        val_loader=loaders["val"],
        schedule=schedule,
        loss_objective=loss_objective,
        _use_default_schedule=use_default,
        generative_kd=config["generative_kd"],
        total_steps=args.total_steps,
        lr=args.lr,
        on_policy_start=config["on_policy_start"],
        quantize=config["quantize"],
        device=args.device,
        rollout_prompts=rollout_prompts,
        reabsorb_every_steps=pra_steps if pra_steps > 0 else None,
    )
    train_time = time.time() - t0
    final_ppl = result.best_metric

    summary = {
        "rung": args.rung,
        "tag": args.tag,
        "config": config,
        "teacher_params_M": teacher_params / 1e6,
        "student_params_M": student_params / 1e6,
        "compression_ratio": teacher_params / max(1, student_params),
        "arch_multiplier": float(arch_mult),
        "target_compression": float(args.target_compression) if args.target_compression else None,
        "reabsorb_every_steps": int(pra_steps),
        "student_hidden_size": int(student.config.n_embd),
        "student_intermediate_size": int(student.config.n_inner or 4 * student.config.n_embd),
        "profile_time_s": profile_time,
        "build_time_s": build_time,
        "train_time_s": train_time,
        "teacher_ppl": teacher_ppl,
        "initial_student_ppl": initial_student_ppl,
        "final_student_ppl": final_ppl,
        "val_kl_forward": result.val_kl_forward,
        "val_kl_reverse": result.val_kl_reverse,
        "total_steps": args.total_steps,
        "behavioral_rank_sample": {
            b.name: int(b.behavioral_rank) for b in profile.branches[:6]
        },
        "variance_rank_sample": {
            b.name: int(b.variance_rank) for b in profile.branches[:6]
        },
    }
    print(f"[fasd-ablation-result] {json.dumps(summary)}")
    _write_json(result_path, summary)


if __name__ == "__main__":
    main()
