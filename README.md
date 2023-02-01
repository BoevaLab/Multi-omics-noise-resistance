# More data, more problems? On noise resistance in multi-omics cancer survival models
## Abstract
With decreasing costs of high-throughput sequencing, more and more studies have started to provide multi-omics molecular profiles of cancer patients. This has led to various developments of novel survival analysis approaches integrating these heterogeneous molecular and clinical groups of variables. While some of these methods have reached state-of-the-art results in cancer survival prediction, many have demonstrated a decay in performance when integrating larger numbers of omics modalities. Here, we address the need to review and further investigate this decay shown by multi-omics integration methods, i.e., assess their modality-level noise resistance. Concretely, we compared eight deep learning and four statistical integration techniques for multi-omics survival data on 17 datasets from The Cancer Genome Atlas (TCGA) in terms of their modality-level noise resistance and general performance. We found that one neural network method, late fusion using the arithmetic mean, and two statistical methods, PriorityLasso and BlockForest, performed best in terms of both noise resistance and overall discriminative and calibration performance. Nevertheless, all tested techniques struggled to adequately handle noise in the form of a high number of modalities and decreased in performance when less informative modalities were added during training. In summary, we confirmed that current multi-omics survival methods are not sufficiently noise-resistant. We recommend relying on only modalities for which there is known predictive value for a particular cancer type until models that have stronger noise-resistance properties are developed.

## Reproducibility
### From scratch
Since installing via conda and R in bash scripts can be finicky and anyway requires user input, we guide you through the process below, after which you may reproduce all of our results by executing our reproduction script.

#### Python
Please run the following in a terminal and give user input as appropriate (e.g., confirming that you want to create a new conda env).

```sh
conda create -n noise_resistance python==3.10.0
conda activate noise_resistance
pip install -r requirements.txt
pip install -e .
```

#### R
We require you to have R 4.1.3 installed - we recommend to use [Rig](https://github.com/r-lib/rig) to manage different R versions.

Supposing you already have R 4.1.3, you may simply run the below in a terminal.

```sh
Rscript -e "install.packages('renv');require(renv);renv::activate();renv::restore()"
```

#### Running experiments
Once both the necessary R and Python packages are installed, you may reproduce all of our work (including data downloads, preprocessing, etc) by running the below in a terminal (make sure to activate the respective conda environment if it is not already active). 

```sh
bash reproduce.sh
```

### Results
All of our results, including preprocessed data, computed performance metrics and predicted survival functions for all models and experiments are available on [Zenodo](https://zenodo.org/record/7529459).

## Caveats
### Speed
Please note that none of the models here (in particular the methods in torch) have been optimised for minimising computation times, so if computation times are important to you, we advise you to reimplement them (especially the Cox loss, following a strategy similar to [1]).

### Code clarity
Please note that several parts of the code could be made clearer through refactoring - we have left them in their original state in order to ensure reproducibility. In case anything is unclear, please reach out.

## Questions
In case of any questions, please reach out to david.wissel@inf.ethz.ch or open an issue in this repo.

## Citation
Our manuscript is still under review.

## References
[1] Simon, Noah, et al. "Regularization paths for Cox’s proportional hazards model via coordinate descent." Journal of statistical software 39.5 (2011): 1.
