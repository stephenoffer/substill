#!/usr/bin/env bash
# Paper-grade benchmark matrix. Experiments ordered by importance so we get
# the most paper-relevant data first if the run is interrupted.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=${EPOCHS:-20}
FT_EPOCHS=${FT_EPOCHS:-6}
THRESHOLDS_CNN=${THRESHOLDS_CNN:-"0.85 0.95"}
ABL_EPOCHS=${ABL_EPOCHS:-15}
ABL_THRESHOLD=${ABL_THRESHOLD:-0.95}

log() { echo; echo "===================================================================="; echo "$@"; echo "===================================================================="; }

# ========================================================================
# 1. ABLATIONS (same teacher already available — fastest, highest priority)
# Each ablation isolates one improvement's contribution to the final method.
# ========================================================================

log "ABLATION 1/7: full ASD"
python scripts/08_ablation.py --variant full --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_full || echo "FAILED (continuing)"

log "ABLATION 2/7: no logit KD"
python scripts/08_ablation.py --variant no_logit_kd --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_no_logit_kd || echo "FAILED (continuing)"

log "ABLATION 3/7: GAP subspace (no spatial)"
python scripts/08_ablation.py --variant gap_subspace --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_gap_subspace || echo "FAILED (continuing)"

log "ABLATION 4/7: GAP covariance (no per-pixel)"
python scripts/08_ablation.py --variant gap_cov --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_gap_cov || echo "FAILED (continuing)"

log "ABLATION 5/7: no sparsity loss"
python scripts/08_ablation.py --variant no_sparsity --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_no_sparsity || echo "FAILED (continuing)"

log "ABLATION 6/7: classical KD (task + logit KD only)"
python scripts/08_ablation.py --variant classical_kd --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_classical_kd || echo "FAILED (continuing)"

log "ABLATION 7/7: task only (no distillation)"
python scripts/08_ablation.py --variant task_only --threshold $ABL_THRESHOLD --epochs $ABL_EPOCHS \
  --output-dir outputs/ablation_task_only || echo "FAILED (continuing)"

# ========================================================================
# 2. CROSS-ARCHITECTURE (CIFAR-10 with different ResNet depths)
# ========================================================================

log "ResNet18 / CIFAR-10"
python scripts/07_bench.py --model resnet18 --dataset cifar10 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet18_cifar10 || echo "FAILED (continuing)"

log "ResNet34 / CIFAR-10"
python scripts/07_bench.py --model resnet34 --dataset cifar10 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet34_cifar10 || echo "FAILED (continuing)"

# ========================================================================
# 3. LLM experiment (generalization beyond CNN)
# ========================================================================

log "GPT-2 / WikiText-2 (τ=0.95)"
python scripts/09_llm_distill.py --threshold 0.95 --epochs 2 --batch-size 8 --seq-len 256 \
  --output-dir outputs/llm || echo "LLM FAILED (continuing)"

# ========================================================================
# 4. Cross-dataset (CIFAR-100)
# ========================================================================

log "ResNet18 / CIFAR-100"
python scripts/07_bench.py --model resnet18 --dataset cifar100 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet18_cifar100 || echo "FAILED (continuing)"

log "ResNet50 / CIFAR-100"
python scripts/07_bench.py --model resnet50 --dataset cifar100 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet50_cifar100 || echo "FAILED (continuing)"

# ========================================================================
# 5. Extra: bigger model (ResNet101) and ResNet34 on CIFAR-100
# ========================================================================

log "ResNet34 / CIFAR-100"
python scripts/07_bench.py --model resnet34 --dataset cifar100 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet34_cifar100 || echo "FAILED (continuing)"

log "ResNet101 / CIFAR-10"
python scripts/07_bench.py --model resnet101 --dataset cifar10 \
  --thresholds $THRESHOLDS_CNN --epochs $EPOCHS --ft-epochs $FT_EPOCHS \
  --output-dir outputs/bench_resnet101_cifar10 || echo "FAILED (continuing)"

log "GPT-2 / WikiText-2 (τ=0.90) — second LLM point"
python scripts/09_llm_distill.py --threshold 0.90 --epochs 2 --batch-size 8 --seq-len 256 \
  --output-dir outputs/llm || echo "LLM FAILED (continuing)"

# ========================================================================
# Aggregate
# ========================================================================

log "Aggregating all results"
python scripts/10_aggregate.py
python scripts/11_compare_sweeps.py --improved outputs/sweep_improved/sweep_results.json \
  --output outputs/paper/baseline_vs_improved.png || echo "compare failed"
echo "Done. See outputs/paper/results_table.md"
