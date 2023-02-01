#!/bin/bash

for fusion in "early"; do
    for modalities in "0" "1" "2" "3" "4" "5" "6"; do
        python -u scripts/python/driver.py --data_dir ./data/ \
            --config_path ./config/ \
            --results_path ./results/ \
            --fusion ${fusion} \
            --modalities ${modalities} \
            --data regular
    done
done

for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
        "intermediate_concat" "intermediate_embrace" \
        "intermediate_attention"; do
    for modalities in "0,1" "0,1,2,3,4,5,6"; do
        python -u scripts/python/driver.py --data_dir ./data/ \
            --config_path ./config/ \
            --results_path ./results/ \
            --fusion ${fusion} \
            --modalities ${modalities} \
            --data regular
    done
done

for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
        "intermediate_concat" "intermediate_embrace" \
        "intermediate_attention"; do
    for modalities in "0,1"; do
        python -u scripts/python/driver.py --data_dir ./data/ \
            --config_path ./config/ \
            --results_path ./results/ \
            --fusion ${fusion} \
            --modalities ${modalities} \
            --data regular
            --target 1
    done
done

for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
        "intermediate_concat" "intermediate_embrace" \
        "intermediate_attention"; do
    for modalities in "0,1" "0,1,2,3,4,5,6"; do
        python -u scripts/python/driver.py --data_dir ./data/ \
            --config_path ./config/ \
            --results_path ./results/ \
            --fusion ${fusion} \
            --modalities ${modalities} \
            --data regular \
            --modality_dropout 1
    done
done

for n_pca_dimensionality in 16 32 64; do
    for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
        "intermediate_concat" "intermediate_embrace" \
        "intermediate_attention"; do
        for modalities in "0,1" "0,1,2,3,4,5,6"; do
            for pca_separate in "separate"; do
                python -u scripts/python/driver.py --data_dir ./data/ \
                    --config_path ./config/ \
                    --results_path ./results/ \
                    --fusion ${fusion} \
                    --modalities ${modalities} \
                    --data pca \
                    --n_pca_dimensionality ${n_pca_dimensionality} \
                    --pca_separate ${pca_separate}
            done
        done
    done
done

for n_noise_dimensionality in 10000; do
    for n_noise_modalities in 1 3 5; do
        for target in 0 1; do
            for fusion in "early" "late_mean" "late_moe" "intermediate_mean" "intermediate_max" \
                "intermediate_concat" "intermediate_embrace" \
                "intermediate_attention"; do
                for modalities in "0,1"; do
                    python -u scripts/python/driver.py --data_dir ./data/ \
                        --config_path ./config/ \
                        --results_path ./results/ \
                        --fusion ${fusion} \
                        --modalities ${modalities} \
                        --data noise \
                        --n_noise_modalities ${n_noise_modalities} \
                        --n_noise_dimensionality ${n_noise_dimensionality} \
                        --target ${target}
                done
            done
        done
    done
done

for modalities in "0" "1" "2" "3" "4" "5" "6"; do
    Rscript scripts/R/driver.R \
        --config_path ./config/ \
        --results_path ./results/ \
        --modalities ${modalities} \
        --data regular
done

for modalities in "0,1" "0,1,2,3,4,5,6"; do
    Rscript scripts/R/driver.R \
        --config_path ./config/ \
        --results_path ./results/ \
        --modalities ${modalities} \
        --data regular
done

for pca_separate in "separate"; do
    for n_pca_dimensionality in 16 32 64; do
        for modalities in "0,1" "0,1,2,3,4,5,6"; do
            Rscript scripts/R/driver.R \
                --config_path ./config/ \
                --results_path ./results/ \
                --modalities ${modalities} \
                --data pca \
                --n_pca_dimensionality ${n_pca_dimensionality} \
                --pca_separate ${pca_separate}
        done
    done
done

for n_noise_dimensionality in 10000; do
    for n_noise_modalities in 1 3 5; do
        for target in 0 1; do
            for modalities in "0,1"; do
                Rscript scripts/R/driver.R \
                    --config_path ./config/ \
                    --results_path ./results/ \
                    --modalities ${modalities} \
                    --data noise \
                    --n_noise_modalities ${n_noise_modalities} \
                    --n_noise_dimensionality ${n_noise_dimensionality} \
                    --target ${target}
            done
        done
    done
done
