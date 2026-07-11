#!/usr/bin/env bash
# Submit the CPSD validation matrix as independent small-GPU (A10G) Anyscale jobs.
#
# Each job runs scripts/fsd/fsd_headline_experiment.py for ONE (compression x seed) cell,
# training the baseline + novel variants head-to-head on GPT-2 + WikiText-2:
#   r2_fasd            absorbed init + KD            (F-ASD baseline)
#   r3_fsd_kd_stiefel  RR-Norm Q trained on Stiefel  (FSD baseline)
#   cpsd_mt            CPI + projection bases trained on Stiefel   (NOVEL: manifold training)
#   cpsd_full          cpsd_mt + differentiable rank (DDR)         (NOVEL: full CPSD)
#
# The decisive question: do cpsd_mt / cpsd_full beat the F-ASD/FSD baselines at matched
# compression? Results -> $ANYSCALE_ARTIFACT_STORAGE/cpsd/$TAG/c<C>_s<SEED>.json
#
# Usage:
#   TAG=cpsd-v1 bash scripts/cpsd/cpsd_experiment_submit.sh
#   CANARY=1 TAG=cpsd-canary bash scripts/cpsd/cpsd_experiment_submit.sh   # 1 quick job
set -eo pipefail

TAG="${TAG:-cpsd-v1}"
TEACHER="${TEACHER:-gpt2}"            # gpt2 (124M) — small/cheap on A10G
STEPS="${STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH="${BATCH:-8}"
LR="${LR:-3e-4}"
CALIB="${CALIB:-32}"
EVAL="${EVAL:-64}"
COMPUTE="${COMPUTE:-asd-gpu-head}"
VARIANTS="${VARIANTS:-r2_fasd,r3_fsd_kd_stiefel,cpsd_mt,cpsd_full}"

if [ "${CANARY:-0}" = "1" ]; then
  COMPRESSIONS="${COMPRESSIONS:-2.0}"
  SEEDS="${SEEDS:-0}"
  STEPS="${STEPS_CANARY:-40}"        # quick sanity run
  echo "[canary] quick verification run: ${STEPS} steps, 2.0x, seed 0"
else
  COMPRESSIONS="${COMPRESSIONS:-2.0 4.0}"
  SEEDS="${SEEDS:-0 1}"
fi

read -ra COMP_ARR <<< "$COMPRESSIONS"
read -ra SEED_ARR <<< "$SEEDS"

echo "Submitting CPSD jobs: TAG=$TAG TEACHER=$TEACHER STEPS=$STEPS"
echo "  compressions: ${COMP_ARR[*]} | seeds: ${SEED_ARR[*]} | variants: $VARIANTS"
echo

for C in "${COMP_ARR[@]}"; do
  for S in "${SEED_ARR[@]}"; do
    NAME="cpsd-${TAG}-c${C}-s${S}"
    OUT="/mnt/cluster_storage/${NAME}.json"
    DST="\$ANYSCALE_ARTIFACT_STORAGE/cpsd/${TAG}/c${C}_s${S}.json"
    echo "  submitting ${NAME}..."
    anyscale job submit \
      --name "${NAME}" \
      --compute-config "${COMPUTE}" \
      --working-dir . \
      --exclude '*.pyc' --exclude '__pycache__' --exclude '.git' \
      --exclude '.pytest_cache' --exclude 'runs' --exclude 'papers' \
      --env "HF_HOME=/mnt/cluster_storage/hf" \
      --env "PYTHONUNBUFFERED=1" \
      -- bash -c "pip install --quiet transformers==5.6.1 datasets==3.6.0 tqdm && \
        PYTHONPATH=. python scripts/fsd/fsd_headline_experiment.py \
          --teacher ${TEACHER} --target-compression ${C} --seed ${S} \
          --steps ${STEPS} --seq-len ${SEQ_LEN} --batch-size ${BATCH} --lr ${LR} \
          --calib-batches ${CALIB} --eval-batches ${EVAL} \
          --variants ${VARIANTS} --output ${OUT} && \
        aws s3 cp ${OUT} ${DST}"
  done
done

echo
echo "All jobs submitted."
echo "Monitor:  anyscale job list | grep cpsd-${TAG}"
echo "Results:  aws s3 ls \$ANYSCALE_ARTIFACT_STORAGE/cpsd/${TAG}/"
