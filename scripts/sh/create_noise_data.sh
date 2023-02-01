#!/bin/bash

for n_noise_dimensionality in 10000; do
    for n_noise_modalities in 1 3 5; do
        for noised_target in 0 1; do
            python -u scripts/python/create_noise_data.py \
                --data_dir ./data/ \
                --config_path ./config/ \
                --modalities "0,1" \
                --n_noise_modalities ${n_noise_modalities} \
                --target ${noised_target} \
                --n_noise_dimensionality ${n_noise_dimensionality}
        done
    done
done
