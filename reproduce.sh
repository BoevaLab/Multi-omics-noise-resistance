#!/bin/bash

bash scripts/sh/download_data.sh &> download_data.log
bash scripts/sh/preprocess_data.sh &> preprocess_data.log
bash scripts/sh/make_splits.sh &> make_splits.log
bash scripts/sh/create_noise_data.sh &> create_noise_data.log
bash scripts/sh/reproduce_experiments.sh &> reproduce_experiments.log
