#!/usr/bin/env bash
# Extended matrix: non-ResNet architectures + SVHN + denser ResNet50/CIFAR-10
# sweep + multi-seed stability checks. Run after the main run_matrix.sh.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=${EPOCHS:-18}   # slightly fewer epochs to fit more cells
FT_EPOCHS=${FT_EPOCHS:-6}

log() { echo; echo "===================================================================="; echo "$@"; echo "===================================================================="; }

# ------------------------------------------------------------------------
# A. Non-ResNet architectures (CIFAR-10)
# ------------------------------------------------------------------------

log "MobileNetV2 / CIFAR-10"
python scripts/13_bench_ext.py --model mobilenet_v2 --dataset cifar10 \
  --thresholds 0.85 0.95 --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_mobilenet_v2_cifar10 || echo "FAILED (continuing)"

log "VGG16-BN / CIFAR-10"
python scripts/13_bench_ext.py --model vgg16_bn --dataset cifar10 \
  --thresholds 0.85 0.95 --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_vgg16_bn_cifar10 || echo "FAILED (continuing)"

# ------------------------------------------------------------------------
# B. SVHN (different task — digit recognition)
# ------------------------------------------------------------------------

log "ResNet18 / SVHN"
python scripts/07_bench.py --model resnet18 --dataset svhn \
  --thresholds 0.85 0.95 --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet18_svhn || echo "FAILED (continuing)"

log "ResNet50 / SVHN"
python scripts/07_bench.py --model resnet50 --dataset svhn \
  --thresholds 0.85 0.95 --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet50_svhn || echo "FAILED (continuing)"

# ------------------------------------------------------------------------
# C. Dense Pareto sweep on ResNet50/CIFAR-10 (lots of points)
# ------------------------------------------------------------------------

log "Dense Pareto curve on ResNet50/CIFAR-10 (9 thresholds)"
python scripts/12_dense_sweep.py --model resnet50 --dataset cifar10 \
  --teacher-weights outputs/teacher_finetuned.pt \
  --thresholds 0.60 0.75 0.80 0.90 0.98 \
  --epochs $EPOCHS \
  --output-dir outputs/dense_sweep_resnet50_cifar10 || echo "FAILED (continuing)"

# ------------------------------------------------------------------------
# D. Multi-seed stability (3 seeds at a fixed τ=0.95)
# ------------------------------------------------------------------------

log "Multi-seed stability: ResNet50/CIFAR-10 @ τ=0.95 (3 seeds)"
python scripts/12_dense_sweep.py --model resnet50 --dataset cifar10 \
  --teacher-weights outputs/teacher_finetuned.pt \
  --thresholds 0.95 --seeds 1 2 3 \
  --epochs $EPOCHS \
  --output-dir outputs/seeds_resnet50_cifar10 || echo "FAILED (continuing)"

# ------------------------------------------------------------------------
# E. Aggregate
# ------------------------------------------------------------------------

log "Aggregating all results"
python scripts/10_aggregate.py
echo "Done. See outputs/paper/results_table.md"
