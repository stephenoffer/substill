#!/usr/bin/env bash
# Submit the F-ASD ablation matrix as independent Anyscale jobs.
# Each job runs one rung of scripts/fasd_ablation.py on one A10G.
#
# v11 (default): matched-compression sweep — every rung at 2x AND 4x — to
# isolate the contribution of Periodic Re-Absorption from compression-ratio
# confounds that polluted v10. Teacher is gpt2-medium.
#
# Results persist to $ANYSCALE_ARTIFACT_STORAGE/fasd/$TAG/results/<rung>-c<C>.json
# (S3-backed via the persistence fix in fasd_ablation.py). Skip-if-exists works
# across cluster restarts.
#
# Usage:
#   TAG=v11-pra-apr29 bash scripts/fasd_ablation_submit.sh
#   TAG=v11-pra-apr29 RUNGS="r1_static r2_pra200" COMPRESSIONS=2.0 bash ...
#   # legacy v10 ladder (8 rungs, no matched compression):
#   TAG=v10-rerun MODE=legacy bash scripts/fasd_ablation_submit.sh

set -eo pipefail

TAG="${TAG:-v11-pra-apr29}"
TEACHER="${TEACHER:-gpt2-medium}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
MODE="${MODE:-v11}"

case "$MODE" in
  v11)
    RUNGS_DEFAULT="r0_random r1_static r2_pra200 r3_pra100 r4_pra200_onpolicy"
    COMPRESSIONS_DEFAULT="2.0 4.0"
    MAX_RANK_DEFAULT=1024  # gpt2-medium hidden = 1024
    ;;
  legacy)
    RUNGS_DEFAULT="0_baseline 1_behavioral 2_procrustes 3_skewkl 4_absorbed 5_onpolicy 6_quantize 7_full"
    COMPRESSIONS_DEFAULT=""  # legacy uses --arch-multiplier, not target-compression
    MAX_RANK_DEFAULT=768
    ;;
  *)
    echo "unknown MODE=$MODE (expected v11 or legacy)" >&2
    exit 1
    ;;
esac

read -ra RUNG_ARRAY <<< "${RUNGS:-$RUNGS_DEFAULT}"
read -ra COMPRESSION_ARRAY <<< "${COMPRESSIONS:-$COMPRESSIONS_DEFAULT}"
MAX_RANK="${MAX_RANK:-$MAX_RANK_DEFAULT}"

# v11 has the rung × compression cross-product; legacy is rung-only.
JOBS=()
if [ "$MODE" = "v11" ]; then
  for C in "${COMPRESSION_ARRAY[@]}"; do
    for R in "${RUNG_ARRAY[@]}"; do
      JOBS+=("${R}|${C}")
    done
  done
else
  for R in "${RUNG_ARRAY[@]}"; do
    JOBS+=("${R}|")
  done
fi

echo "Submitting ${#JOBS[@]} Anyscale jobs"
echo "  TAG=${TAG}"
echo "  TEACHER=${TEACHER}"
echo "  MODE=${MODE}"
echo "  TOTAL_STEPS=${TOTAL_STEPS}"
echo

for SPEC in "${JOBS[@]}"; do
  RUNG="${SPEC%|*}"
  COMPRESSION="${SPEC#*|}"
  if [ -n "$COMPRESSION" ]; then
    NAME="fasd-${TAG}-${RUNG}-c${COMPRESSION}"
    EXTRA_ARGS="--target-compression ${COMPRESSION}"
    EXTRA_ENV="--env FASD_TARGET_COMPRESSION=${COMPRESSION}"
  else
    NAME="fasd-${TAG}-${RUNG}"
    EXTRA_ARGS=""
    EXTRA_ENV=""
  fi

  echo "  submitting ${NAME}..."
  anyscale job submit \
    --name "${NAME}" \
    --compute-config asd-gpu-head \
    --working-dir . \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.git' \
    --exclude '.pytest_cache' --exclude '*.profile' --exclude '*.fasd' \
    --env "FASD_TAG=${TAG}" \
    --env "FASD_RUNG=${RUNG}" \
    ${EXTRA_ENV} \
    --env "HF_HOME=/mnt/cluster_storage/hf" \
    --env "PYTHONUNBUFFERED=1" \
    -- bash -c "pip install --quiet transformers==5.6.1 datasets==3.6.0 tqdm && python scripts/fasd_ablation.py --rung '${RUNG}' --tag '${TAG}' --teacher ${TEACHER} --batch-size 4 --seq-len 128 --total-steps ${TOTAL_STEPS} --rank-tol 0.02 --max-rank ${MAX_RANK} --calib-batches 32 ${EXTRA_ARGS}"
done

echo
echo "All jobs submitted."
echo "Monitor:  anyscale job list --include-all-users | grep fasd-${TAG}"
echo "Results:  aws s3 ls \$ANYSCALE_ARTIFACT_STORAGE/fasd/${TAG}/results/"
