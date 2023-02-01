#!/bin/bash

mkdir -p ./data/processed
mkdir -p ./data/processed/TCGA

Rscript scripts/R/run_preprocessing.R
