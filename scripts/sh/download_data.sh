#!/bin/bash

mkdir -p ./data
mkdir -p ./data/raw

# Download everything except CNV from PANCANATLAS
for id in "d82e2c44-89eb-43d9-b6d3-712732bf6a53" \
    "fcbb373e-28d4-4818-92f3-601ede3da5e1" \
    "3586c0da-64d0-4b74-a449-5ff4d9136611" \
    "1c8cfe5f-e52d-41ba-94da-f15ea1337efc" \
    "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81" \
    "1c6174d9-8ffb-466e-b5ee-07b204c15cf8" \
    "0fc78496-818b-4896-bd83-52db1f533c5c"; do
    wget --content-disposition http://api.gdc.cancer.gov/data/${id} -P ./data/raw/
    sleep 60
done

# Download CNV called with GISTIC2.0 from Xena
wget "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz" -P ./data/raw/
