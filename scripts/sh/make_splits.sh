#!/bin/bash

mkdir -p ./data/splits

python -u scripts/python/rerun_splits.py \
    --data_dir ./data/ \
    --config_path ./config/
