#!/usr/bin/env bash
set -euo pipefail

python run.py \
  -m camfnet_aml \
  -d mustardpp \
  -c cafnet_aml/config/config_regression.json \
  -s 1111 -s 1112 -s 1113 -s 1114 -s 1115 \
  --model-save-dir saved_models \
  --res-save-dir results \
  --log-dir logs \
  -n 0 \
  -v 1
