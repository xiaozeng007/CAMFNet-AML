#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-mult}"
DATASET_NAME="${2:-mustardpp}"

python run.py \
  -m "${MODEL_NAME}" \
  -d "${DATASET_NAME}" \
  -c cafnet_aml/config/config_regression.json \
  -s 1111 -s 1112 -s 1113 \
  --model-save-dir "saved_models/baselines/${MODEL_NAME}" \
  --res-save-dir "results/baselines/${MODEL_NAME}" \
  --log-dir "logs/baselines/${MODEL_NAME}" \
  -n 0 \
  -v 1
