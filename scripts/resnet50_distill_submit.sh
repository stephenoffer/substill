#!/usr/bin/env bash
# Submit the ResNet50 vision benchmark as independent A10G Anyscale jobs (the non-LLM arm).
#
# Each job runs scripts/resnet50_distill.py for ONE width-ratio on CIFAR-10: it warms up the
# teacher's head (ImageNet backbone -> real CIFAR classifier), then builds a channel-narrowed
# student TWO ways at matched compression — random-init vs absorbed-init (FASD) — distils both
# on class logits, and reports top-1. The absorbed-vs-random gap is the vision analogue of the
# GPT-2 "concrete win vs naive baseline".
#
# Results -> $ANYSCALE_ARTIFACT_STORAGE/resnet50/$TAG/...
#
# Usage:
#   TAG=rn50-v1 bash scripts/resnet50_distill_submit.sh
#   CANARY=1 TAG=rn50-canary bash scripts/resnet50_distill_submit.sh
set -eo pipefail

TAG="${TAG:-rn50-v1}"
STEPS="${STEPS:-2000}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-64}"
HEAD_WARMUP="${HEAD_WARMUP:-300}"
COMPUTE="${COMPUTE:-asd-gpu-head}"

if [ "${CANARY:-0}" = "1" ]; then
  RATIOS="${RATIOS:-0.5}"; STEPS="${STEPS_CANARY:-30}"; HEAD_WARMUP=20
  echo "[canary] quick run: ${STEPS} steps, ratio 0.5"
else
  RATIOS="${RATIOS:-0.5 0.35}"
fi
read -ra RATIO_ARR <<< "$RATIOS"

echo "Submitting resnet50 jobs: TAG=$TAG STEPS=$STEPS"
for R in "${RATIO_ARR[@]}"; do
  NAME="rn50-${TAG}-r${R}"
  OUT="/mnt/cluster_storage/${NAME}.json"
  DST="\$ANYSCALE_ARTIFACT_STORAGE/resnet50/${TAG}/r${R}.json"
  echo "  submitting ${NAME}..."
  anyscale job submit \
    --name "${NAME}" \
    --compute-config "${COMPUTE}" \
    --working-dir . \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.git' \
    --exclude '.pytest_cache' --exclude 'runs' --exclude 'papers' \
    --env "HF_HOME=/mnt/cluster_storage/hf" \
    --env "PYTHONUNBUFFERED=1" \
    -- bash -c "pip install --quiet torchvision tqdm && \
      PYTHONPATH=. python scripts/resnet50_distill.py \
        --dataset cifar10 --width-ratio ${R} --steps ${STEPS} --lr ${LR} \
        --batch-size ${BATCH} --head-warmup-steps ${HEAD_WARMUP} --output ${OUT} && \
      aws s3 cp ${OUT} ${DST}"
done

echo
echo "All jobs submitted."
echo "Monitor:  anyscale job list | grep rn50-${TAG}"
echo "Results:  aws s3 ls \$ANYSCALE_ARTIFACT_STORAGE/resnet50/${TAG}/"
