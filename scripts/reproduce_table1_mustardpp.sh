#!/usr/bin/env bash
set -euo pipefail

echo "[CAMFNet-AML] Reproducing Table 1 on MUStARD++"

python run.py \
  -m camfnet_aml \
  -d mustardpp \
  -c cafnet_aml/config/config_regression.json \
  -s 1111 -s 1112 -s 1113 -s 1114 -s 1115 \
  --model-save-dir saved_models/table1_mustardpp \
  --res-save-dir results/table1_mustardpp \
  --log-dir logs/table1_mustardpp \
  -n 0 \
  -v 1

echo "[Done] Check results/table1_mustardpp/normal/mustardpp.csv"
