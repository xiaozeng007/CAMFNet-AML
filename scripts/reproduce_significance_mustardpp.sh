#!/usr/bin/env bash
set -euo pipefail

echo "[CAMFNet-AML] Running paired significance tests on MUStARD++"

python scripts/run_significance_test.py \
  --results-csv results_sig/normal/mustardpp.csv \
  --model-a camfn \
  --model-b muvac \
  --num-permutations 50000 \
  --num-bootstrap 50000 \
  --seed 2026 \
  --output results_sig/statistics/mustardpp_camfn_vs_muvac_sig_10seeds.json

echo "[Done] Check results_sig/statistics/mustardpp_camfn_vs_muvac_sig_10seeds.json"
