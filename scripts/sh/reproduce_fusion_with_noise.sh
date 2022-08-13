#!/bin/bash

for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
    "intermediate_concat" "intermediate_embrace" \
    "intermediate_attention"; do
    for modalities in "0,1"; do
        for n_noise_modalities in 1 3 5; do
            for n_noise_dimensionality in 10000; do
                for noised_target in true false; do
                    python scripts/python/driver.py --data_dir ./data/ \
                        --config_path ./config/ \
                        --results_path ./results/ \
                        --fusion ${fusion} \
                        --modalities ${modalities} \
                        --n_noise_modalities=${n_noise_modalities} \
                        --noised_target=${noised_target} \
                        --n_noise_dimensionality=${n_noise_dimensionality}
                done
            done
        done
    done
done
