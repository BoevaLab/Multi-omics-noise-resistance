# Multi-omics noise resistance
## Project overview
In this project, we were interested in answering two main questions:

- Which neural fusion technique should be used to integrate multi-omics data in a cancer survival setting (our goal here was a de-facto empirical comparison in which we held all factors but the integration technique constant)?
- Which neural fusion technique is most resistant to noise on the modality level (i.e., which integration method does not decay in performance as you add hypothetically useless additional modalities)?

This repo contains code to reproduce all of our experiments, in which we evaluated eight neural multi-omics integration techniques on 17 cancers from TCGA. For further details on the project, please refer to our manuscript.

## Reproducibility
To reproduce all of our results, please first download our data and splits [here](https://drive.google.com/file/d/1gQkLW-UeWZYylFB08f0TuNw5v8MNA2iL/view?usp=sharing), unzip them and place them in the main repo folder (the resulting folder should be called `data`).

Afterward, you may simply run:

```sh
source setup.sh
```

You may then reproduce all of our experiments by running:

```sh
bash run_all.sh
```

Please note that due to potential differences in operating systems or CPU model, you may receive results differing very slightly from our own (since `torch` does not guarantee perfect reproducibility in these cases).

## Questions
In case of any questions, please reach out to david.wissel@inf.ethz.ch or open an issue in this repo.

## Citation
Our manuscript is still under review.
