#!/usr/bin/env bash
set -euo pipefail

echo "[CAMFNet-AML] Reproducing on CMU-MOSEI"

python run.py \
  -m camfnet_aml \
  -d mosei \
  -c cafnet_aml/config/config_regression.json \
  -s 1111 -s 1112 -s 1113 \
  --model-save-dir saved_models/mosei \
  --res-save-dir results/mosei \
  --log-dir logs/mosei \
  -n 0 \
  -v 1

echo "[Done] Check results/mosei/normal/mosei.csv"
