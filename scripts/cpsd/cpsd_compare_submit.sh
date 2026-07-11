#!/usr/bin/env bash
# Submit the matched-compression head-to-head comparison as independent A10G Anyscale jobs.
#
# Each job runs scripts/cpsd/cpsd_compare.py for ONE (arch-multiplier x seed) cell on
# GPT-2 + WikiText-2, training all variants head-to-head:
#   r1_kd_random  random-init + KD          (naive competition floor)
#   r2_fasd       absorbed-init + KD        (prior-art baseline)
#   cpsd_mt       + manifold-trained factors (NOVEL)
#   cpsd_full     + KD-driven differentiable rank (MT+DDR)         (NOVEL, ours)
#   dobi_svd      same pipeline, reconstruction-driven rank        (Dobi-SVD foil)
#
# Decisive question: does cpsd_full (KD-driven rank) beat dobi_svd (reconstruction-driven)
# at matched compression? Results -> $ANYSCALE_ARTIFACT_STORAGE/cpsd_compare/$TAG/...
#
# Usage:
#   TAG=cmp-v1 bash scripts/cpsd/cpsd_compare_submit.sh
#   CANARY=1 TAG=cmp-canary bash scripts/cpsd/cpsd_compare_submit.sh   # 1 quick job
set -eo pipefail

TAG="${TAG:-cmp-v1}"
TEACHER="${TEACHER:-gpt2}"
STEPS="${STEPS:-500}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-4}"
CALIB="${CALIB:-16}"
EVAL="${EVAL:-64}"
COMPUTE="${COMPUTE:-asd-gpu-head}"
VARIANTS="${VARIANTS:-r1_kd_random r2_fasd cpsd_mt cpsd_full dobi_svd}"

# (arch_multiplier, target-compression label) pairs — ~2x and ~3x on GPT-2.
if [ "${CANARY:-0}" = "1" ]; then
  MULTS=("0.5:2"); SEEDS="${SEEDS:-0}"; STEPS="${STEPS_CANARY:-30}"
  echo "[canary] quick run: ${STEPS} steps, mult 0.5, seed 0"
else
  # MULTS override (space-separated "mult:label" pairs), e.g. MULTS="0.5:2".
  read -ra MULTS <<< "${MULTS:-0.5:2 0.35:3}"; SEEDS="${SEEDS:-0 1 2}"
fi
read -ra SEED_ARR <<< "$SEEDS"

echo "Submitting cpsd_compare jobs: TAG=$TAG TEACHER=$TEACHER STEPS=$STEPS"
for pair in "${MULTS[@]}"; do
  MULT="${pair%%:*}"; CLABEL="${pair##*:}"
  for S in "${SEED_ARR[@]}"; do
    NAME="cmp-${TAG}-c${CLABEL}-s${S}"
    OUT="/mnt/cluster_storage/${NAME}.json"
    DST="\$ANYSCALE_ARTIFACT_STORAGE/cpsd_compare/${TAG}/c${CLABEL}_s${S}.json"
    echo "  submitting ${NAME} (arch_mult=${MULT})..."
    anyscale job submit \
      --name "${NAME}" \
      --compute-config "${COMPUTE}" \
      --working-dir . \
      --exclude '*.pyc' --exclude '__pycache__' --exclude '.git' \
      --exclude '.pytest_cache' --exclude 'runs' --exclude 'papers' \
      --env "HF_HOME=/mnt/cluster_storage/hf" \
      --env "PYTHONUNBUFFERED=1" \
      -- bash -c "pip install --quiet transformers==5.6.1 datasets==3.6.0 tqdm && \
        PYTHONPATH=. python scripts/cpsd/cpsd_compare.py \
          --teacher ${TEACHER} --arch-multiplier ${MULT} --target-compression ${CLABEL} \
          --seed ${S} --steps ${STEPS} --seq-len ${SEQ_LEN} --batch-size ${BATCH} --lr ${LR} \
          --calib-batches ${CALIB} --eval-batches ${EVAL} \
          --variants ${VARIANTS} --output ${OUT} && \
        aws s3 cp ${OUT} ${DST}"
  done
done

echo
echo "All jobs submitted."
echo "Monitor:  anyscale job list | grep cmp-${TAG}"
echo "Results:  aws s3 ls \$ANYSCALE_ARTIFACT_STORAGE/cpsd_compare/${TAG}/"
echo "Aggregate (after sync): python scripts/cpsd/cpsd_aggregate.py results/*.json"
