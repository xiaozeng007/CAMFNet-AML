#!/usr/bin/env bash
set -euo pipefail

echo "[CAMFNet-AML] Running conflict ablation on MUStARD++"

python scripts/run_camfn_conflict_ablation.py \
  --seeds 1111,1112,1113,1114,1115 \
  --gpu-id 0 \
  --num-workers 0 \
  --root results/ablation/conflict \
  --key-metric Accuracy

echo "[Done] Check results/ablation/conflict/mustardpp/summary_conflict_ablation.csv"
