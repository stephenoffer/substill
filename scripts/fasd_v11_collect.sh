#!/usr/bin/env bash
# Wait for the v11 ladder to finish, then pull results from S3 and emit a
# ranked summary. Driver script for the autolaunched fasd-v11-pra-apr30 ladder.
#
# Usage: LADDER_TAG=v11-pra-apr30 bash fasd_v11_collect.sh

set -uo pipefail

LADDER_TAG="${LADDER_TAG:-v11-pra-apr30}"
POLL_SECS="${POLL_SECS:-180}"
RESULTS_S3="${ANYSCALE_ARTIFACT_STORAGE}/fasd/${LADDER_TAG}/results"
LOCAL_OUT="/tmp/fasd_${LADDER_TAG}"
SUMMARY_MD="${LOCAL_OUT}/SUMMARY.md"

JOB_IDS=(
  "fasd-v11-pra-apr30-r0_random-c2.0|prodjob_ibybuen6u3gu3qhnvjyf49r8ip"
  "fasd-v11-pra-apr30-r1_static-c2.0|prodjob_dk2zjj4dqiij139n8guaxqukxi"
  "fasd-v11-pra-apr30-r2_pra200-c2.0|prodjob_h3467b8acq8h6tkv7cfjwjhjy7"
  "fasd-v11-pra-apr30-r3_pra100-c2.0|prodjob_al6weyjqpz1qm5xsvgmutkalts"
  "fasd-v11-pra-apr30-r4_pra200_onpolicy-c2.0|prodjob_ybqrceh7ksveueibln8694hguy"
  "fasd-v11-pra-apr30-r0_random-c4.0|prodjob_8rbkgpsuk3f1wk39n26k8ukupy"
  "fasd-v11-pra-apr30-r1_static-c4.0|prodjob_97r16grku7il7myavv11amlfzr"
  "fasd-v11-pra-apr30-r2_pra200-c4.0|prodjob_e8zp7dvkcw7xdllp18q3bim8ac"
  "fasd-v11-pra-apr30-r3_pra100-c4.0|prodjob_i5ldjfnzhx3wqwntghk3pnkhtu"
  "fasd-v11-pra-apr30-r4_pra200_onpolicy-c4.0|prodjob_b2wccpeq9kwql15yh2bj39g2p3"
)

mkdir -p "${LOCAL_OUT}"
echo "[collect] tag=${LADDER_TAG}  poll=${POLL_SECS}s  jobs=${#JOB_IDS[@]}"

is_terminal() {
  case "$1" in
    SUCCEEDED|SUCCESS|FAILED|BROKEN|TERMINATED|OUT_OF_RETRIES) return 0;;
    *) return 1;;
  esac
}

while true; do
  TS="$(date +%H:%M:%S)"
  PENDING=0
  declare -A STATES
  STATE_LINE=""
  for SPEC in "${JOB_IDS[@]}"; do
    NAME="${SPEC%|*}"
    ID="${SPEC#*|}"
    STATE="$(anyscale job status --job-id "${ID}" 2>/dev/null \
              | awk -F': *' '/^state:/ {print $2; exit}')"
    STATE="${STATE:-UNKNOWN}"
    STATES["${NAME}"]="${STATE}"
    if ! is_terminal "${STATE}"; then PENDING=$((PENDING+1)); fi
    STATE_LINE+=" ${NAME##*-}=${STATE}"
  done
  echo "[collect ${TS}] pending=${PENDING}/${#JOB_IDS[@]} ${STATE_LINE}"
  if [ "${PENDING}" -eq 0 ]; then
    echo "[collect] all jobs terminal, pulling results"
    break
  fi
  sleep "${POLL_SECS}"
done

aws s3 sync --quiet "${RESULTS_S3}/" "${LOCAL_OUT}/results/" || true
echo "[collect] downloaded results:"
ls -la "${LOCAL_OUT}/results/" || true

python3 - "${LOCAL_OUT}/results" "${SUMMARY_MD}" "${LADDER_TAG}" <<'PY'
import json, os, sys
from pathlib import Path

results_dir, out_md, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for p in sorted(Path(results_dir).glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f"skip {p}: {e}"); continue
    rows.append(d)

if not rows:
    Path(out_md).write_text(f"# v11 summary ({tag})\n\nNo results found.\n")
    print("no results"); sys.exit(0)

def f(x, n=2):
    if x is None: return "—"
    try: return f"{x:.{n}f}"
    except Exception: return str(x)

# Group by target compression
buckets = {}
for r in rows:
    c = r.get("target_compression") or "?"
    buckets.setdefault(c, []).append(r)

lines = [f"# v11 ladder results ({tag})", ""]
lines.append(f"Teacher: {rows[0].get('teacher_params_M','?')}M params  "
             f"| teacher PPL: {f(rows[0].get('teacher_ppl'))}")
lines.append("")
for c in sorted(buckets):
    lines.append(f"## Target compression {c}×")
    lines.append("")
    lines.append("| Rung | actual× | student M | initPPL | finalPPL | KLfwd | KLrev | profile s | train s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(buckets[c], key=lambda x: x.get("final_student_ppl", float("inf"))):
        lines.append(
            f"| `{r.get('rung','?')}` "
            f"| {f(r.get('compression_ratio'))} "
            f"| {f(r.get('student_params_M'))} "
            f"| {f(r.get('initial_student_ppl'),1)} "
            f"| **{f(r.get('final_student_ppl'),1)}** "
            f"| {f(r.get('val_kl_forward'),3)} "
            f"| {f(r.get('val_kl_reverse'),3)} "
            f"| {f(r.get('profile_time_s'),0)} "
            f"| {f(r.get('train_time_s'),0)} |"
        )
    lines.append("")

# Cross-comparison: PRA delta vs r1_static at each compression
lines.append("## PRA effect (Δ final-PPL vs r1_static within same compression)")
lines.append("")
lines.append("| compression | r1_static PPL | r2_pra200 Δ | r3_pra100 Δ | r4_pra200_onpolicy Δ | r0_random Δ |")
lines.append("|---|---|---|---|---|---|")
for c in sorted(buckets):
    by_rung = {r.get("rung"): r for r in buckets[c]}
    base = by_rung.get("r1_static")
    if not base or base.get("final_student_ppl") is None:
        lines.append(f"| {c}× | (no baseline) | — | — | — | — |")
        continue
    bp = base["final_student_ppl"]
    def delta(rung):
        r = by_rung.get(rung)
        if not r or r.get("final_student_ppl") is None: return "—"
        d = r["final_student_ppl"] - bp
        sign = "↓" if d < 0 else ("↑" if d > 0 else "=")
        return f"{sign}{abs(d):.1f}"
    lines.append(
        f"| {c}× | {f(bp,1)} "
        f"| {delta('r2_pra200')} "
        f"| {delta('r3_pra100')} "
        f"| {delta('r4_pra200_onpolicy')} "
        f"| {delta('r0_random')} |"
    )

Path(out_md).write_text("\n".join(lines) + "\n")
print(f"wrote {out_md}")
print()
print("\n".join(lines))
PY

# Mirror summary to S3 for easy sharing
aws s3 cp --quiet "${SUMMARY_MD}" "${ANYSCALE_ARTIFACT_STORAGE}/fasd/${LADDER_TAG}/SUMMARY.md" || true
echo
echo "[collect] summary at ${SUMMARY_MD}"
echo "[collect] mirrored to ${ANYSCALE_ARTIFACT_STORAGE}/fasd/${LADDER_TAG}/SUMMARY.md"
