#!/bin/bash

for fusion in "early_ae" "intermediate_ae"; do
    echo "Starting ${fusion}"
    for modalities in "0,1" "0,1,2,3,4,5,6"; do
        echo "Starting ${modalities}"
        python scripts/python/driver_ae.py --data_dir ./data/ \
            --config_path ./config/ \
            --results_path ./results/ \
            --fusion ${fusion} \
            --modalities ${modalities} \
            --n_noise_modalities 0 --noised_target false
    done
done
