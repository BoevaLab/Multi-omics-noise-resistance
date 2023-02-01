run_benchmark <- function(
    config_path,
    results_path,
    modalities,
    input="regular",
    n_noise_modalities=0,
    target=FALSE,
    n_noise_dimensionality=0,
    n_pca_dimensionality=0,
    pca_separate=TRUE
) {
    suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3pipelines)
  library(mlr3learners)
  library(mlr3tuning)
  library(paradox)
  library(mlr3proba)
  library(rjson)
  library(dplyr)
  library(forcats)
  source(here::here("noise_resistance", "R", "learners", "blockforest_learner.R"))
  source(here::here("noise_resistance", "R", "learners", "cv_glmnet_learner.R"))
  source(here::here("noise_resistance", "R", "learners", "cv_prioritylasso_learner.R"))
})

# Set up parallelisation option.
future::plan("multisession")
options(future.globals.onReference = "ignore")

config <- rjson::fromJSON(
  file = here::here(config_path, "config.json")
)

# Set up mlr3 pipelines.
remove_constants <- po("removeconstants")
encode <- po("encode", method = "treatment")
fix_factors <- po("fixfactors")
impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))
imp_cat <- po("imputeoor",
              affect_columns = selector_grep("clinical"))

pipe <- remove_constants %>>% fix_factors
pipe_ohe <- pipe %>>% encode %>>% impute
# Seeding for reproducibility.
set.seed(config$random_seed)


if (length(modalities) == 1) {
  model_names <- c(
  "Lasso",
  "RSF"
)

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
learners <- list(
  pipe_ohe %>>% po("learner",
    id = "lasso",
    learner = lrn("surv.cv_glmnet_custom",
      fallback = lrn("surv.kaplan"),
      s = "lambda.min", standardize = TRUE, nfolds = 5, alpha = 0.95
    )
  ),
    pipe %>>% imp_cat %>>% po("learner",
    id = "rsf",
    learner = lrn("surv.ranger",
      num.trees = 2000,
      splitrule = "extratrees"
    )
  )

)
}

# PCA
else if (input == "pca") {
  if (length(modalities) == 2) {
pca = po("scale") %>>% po("pca", id = "gex_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^gex")) %>>%
  po("renamecolumns", renaming = setNames(paste0("gex_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "gex_rename")
  }
  else {
pca = po("scale") %>>% po("pca", id = "gex_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^gex")) %>>%
  po("renamecolumns", renaming = setNames(paste0("gex_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "gex_rename") %>>%
  po("pca", id = "rppa_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^rppa")) %>>%
  po("renamecolumns", renaming = setNames(paste0("rppa_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "rppa_rename") %>>%
  po("pca", id = "mirna_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^mirna")) %>>%
  po("renamecolumns", renaming = setNames(paste0("mirna_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "mirna_rename") %>>%
  po("pca", id = "mutation_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^mutation")) %>>%
  po("renamecolumns", renaming = setNames(paste0("mutation_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "mutation_rename") %>>%
  po("pca", id = "meth_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^meth")) %>>%
  po("renamecolumns", renaming = setNames(paste0("meth_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "meth_rename") %>>%
  po("pca", id = "cnv_pca", rank. = n_pca_dimensionality, affect_columns = selector_grep("^cnv")) %>>%
  po("renamecolumns", renaming = setNames(paste0("cnv_PC", 1:n_pca_dimensionality), paste0("PC", 1:n_pca_dimensionality)), id = "cnv_rename")



    
  }
    graph <- pipe %>>% pca
    graph_ohe <- pipe_ohe %>>% pca

    pipe <- remove_constants %>>% fix_factors
    pipe_ohe <- pipe %>>% encode %>>% impute
  model_names <- c(
  "BlockForest",
  "RSF",
  "prioritylasso",
  "Lasso"
)

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
learners <- list(
  pipe %>>% imp_cat %>>% pca %>>% po("learner",
    id = "blockforest",
    learner = lrn("surv.blockforest",
      fallback = lrn("surv.kaplan"),
      block.method = "BlockForest",
      num.trees = 2000, mtry = NULL, nsets = config$bf_tuning_rounds, num.trees.pre = 1500,
      splitrule = "extratrees"
    )
  ),
  pipe %>>% imp_cat %>>% pca %>>% po("learner",
    id = "rsf",
    learner = lrn("surv.ranger",
      fallback = lrn("surv.kaplan"),
      num.trees = 2000,
      splitrule = "extratrees"
    )
  ),
  pipe_ohe %>>% pca %>>% po("learner",
    id = "prioritylasso",
    learner = lrn("surv.cv_prioritylasso",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block1.penalization = TRUE, lambda.type = "lambda.min",
      standardize = FALSE, 
      nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5
    )
  ),
  pipe_ohe %>>% pca %>>% po("learner",
    id = "lasso",
    learner = lrn("surv.cv_glmnet_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      s = "lambda.min", standardize = FALSE, nfolds = 5
    )
  )
)
}

else {
# Model names to be reproduced.
model_names <- c(
  "BlockForest",
  "RSF",
  "prioritylasso",
  "Lasso"
)

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
learners <- list(
  pipe %>>% imp_cat %>>% po("learner",
    id = "blockforest",
    learner = lrn("surv.blockforest",
      block.method = "BlockForest",
      num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 100,
      splitrule = "extratrees"
    )
  ),
  pipe %>>% imp_cat %>>% po("learner",
    id = "rsf",
    learner = lrn("surv.ranger",
      num.trees = 2000,
      splitrule = "extratrees"
    )
  ),
  pipe_ohe %>>% po("learner",
    id = "prioritylasso",
    learner = lrn("surv.cv_prioritylasso",
      fallback = lrn("surv.kaplan"),
      block1.penalization = TRUE, lambda.type = "lambda.min",
      standardize = TRUE, 
      nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, alpha = 0.95
    )
  ),
  pipe_ohe %>>% po("learner",
    id = "lasso",
    learner = lrn("surv.cv_glmnet_custom",
      fallback = lrn("surv.kaplan"),
      s = "lambda.min", standardize = TRUE, nfolds = 5, alpha = 0.95
    )
  )

)
}


# Iterate over all cancers in the "TCGA".
for (cancer in config["datasets"]) {
# Read in complete modality sample dataset.
  data <- vroom::vroom(
    here::here(
    "data", "processed", "TCGA",
    paste0(cancer, "_data_preprocessed.csv", collapse = "")
    ))
if (input == "noise") {
  if (target) {
    without_string <- "with_target"
  }
  else {
    without_string <- "without_target"
  }
  data <- vroom::vroom(
    here::here(
    "data", "processed", "TCGA",
     "noise", n_noise_modalities, without_string, paste0(n_noise_dimensionality, "_noise_dimensionality"), paste0(paste0(modalities, collapse = "_"), "_preprocessed.csv"))
)
}

else if (input == "pca") {
    data <- vroom::vroom(
    here::here(
    "data", "processed", "TCGA",
     "PCA", 
     pca_separate,
    paste0(cancer, "_PCA_", min(n_pca_dimensionality, ncol(data) - 3), "_preprocessed.csv") 
    )
)

}

else if (target) {
  data$clinical_target <- data$OS_days
}

mask <- sapply(strsplit(colnames(data), "_"), function(x) x[[1]]) %in% c(modalities, "OS")
data <- data[, mask]
# Remove patient_id column and explicitly cast character columns as strings.
data <- data %>% mutate(across(where(is.character), as.factor))
# Create mlr3 task dataset - our event time is indicated by `OS_days`
# and our event by `OS`. All our datasets are right-censored.

tmp <- as_task_surv(data.frame(data),
    time = "OS_days",
    event = "OS",
    type = "right",
    id = cancer
)
format_splits <- function(raw_splits) {
  if (any(is.na(raw_splits))) {
    apply(data.frame(raw_splits), 1, function(x) unname(x[!is.na(x)]) + 1)
  } else {
    x <- unname(as.matrix(raw_splits)) + 1
    split(x, row(x))
  }
}
# Add stratification on the event - this is only necessary if you
# use mlr3 to tune hyperparameters, but it is good to keep in for
# safety.
tmp$add_strata("OS")
# Iterate over `get_splits` to get full train and test splits for usage in mlr3.
train_splits <- format_splits(readr::read_csv(here::here(
  "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
)))
test_splits <- format_splits(readr::read_csv(here::here(
  "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
)))
# Run benchmark using mlr3.
bmr <- benchmark(benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
))
# Score benchmark such that we get access to prediction objects.
bmr <- bmr$score()
# Loop over and write out predictions for all models (zero-indexed,
# since we need this for getting the right survival functions).
for (model in 0:(length(model_names) - 1)) {
  if (input == "regular") {
    if (!target) {
    if (!dir.exists(here::here(
    results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)], paste0(modalities, collapse="_")
    ))) {
    dir.create(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)],  paste0(modalities, collapse="_")
        ),
        recursive = TRUE
    )
    }
    # Write out CSV file of survival function prediction for each split.
    for (i in 1:(config$n_outer_repetitions * config$n_outer_splits)) {
    data.frame(bmr[((1 + (model * (config$n_outer_repetitions * config$n_outer_splits))):((model + 1) * (config$n_outer_repetitions * config$n_outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
        here::here(
        results_path, "survival_functions", "TCGA", cancer,  model_names[(model + 1)], paste0(modalities, collapse="_"), paste0("split_", i, ".csv")
        )
    )
    }
    }
    else {
          if (!dir.exists(here::here(
    results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)], paste0(modalities, collapse="_"), "with_target"
    ))) {
    dir.create(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)],  paste0(modalities, collapse="_"), "with_target"
        ),
        recursive = TRUE
    )
    }
    # Write out CSV file of survival function prediction for each split.
    for (i in 1:(config$n_outer_repetitions * config$n_outer_splits)) {
    data.frame(bmr[((1 + (model * (config$n_outer_repetitions * config$n_outer_splits))):((model + 1) * (config$n_outer_repetitions * config$n_outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
        here::here(
        results_path, "survival_functions", "TCGA", cancer,  model_names[(model + 1)], paste0(modalities, collapse="_"), "with_target", paste0("split_", i, ".csv")
        )
    )
    }
    }

  }
  else if (input == "noise") {
        if (!dir.exists(here::here(
    results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)], paste0(modalities, collapse="_"), "noise", paste0(n_noise_modalities, "_noise_modalities_", without_string), paste0(n_noise_dimensionality, "_noise_dimensions")
    ))) {
    dir.create(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)], paste0(modalities, collapse="_"), "noise", paste0(n_noise_modalities, "_noise_modalities_", without_string), paste0(n_noise_dimensionality, "_noise_dimensions")
        ),
        recursive = TRUE
    )
    }
    # Write out CSV file of survival function prediction for each split.
    for (i in 1:(config$n_outer_repetitions * config$n_outer_splits)) {
    data.frame(bmr[((1 + (model * (config$n_outer_repetitions * config$n_outer_splits))):((model + 1) * (config$n_outer_repetitions * config$n_outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)], paste0(modalities, collapse="_"), "noise", paste0(n_noise_modalities, "_noise_modalities_", without_string), paste0(n_noise_dimensionality, "_noise_dimensions"), paste0("split_", i, ".csv")
        )
    )
    }
  }

  else if (input == "pca") {
    if (!dir.exists(here::here(
    results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)],  paste0(modalities, collapse="_"), "PCA", pca_separate, n_pca_dimensionality
    ))) {
    dir.create(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)],  paste0(modalities, collapse="_"), "PCA", pca_separate, n_pca_dimensionality
        ),
        recursive = TRUE
    )
    }
    # Write out CSV file of survival function prediction for each split.
    for (i in 1:(config$n_outer_repetitions * config$n_outer_splits)) {
    data.frame(bmr[((1 + (model * (config$n_outer_repetitions * config$n_outer_splits))):((model + 1) * (config$n_outer_repetitions * config$n_outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
        here::here(
        results_path, "survival_functions", "TCGA", cancer, model_names[(model + 1)],  paste0(modalities, collapse="_"), "PCA", pca_separate, n_pca_dimensionality, paste0("split_", i, ".csv")
        )
    )
    }
  }

}
}
}

suppressPackageStartupMessages({
  library(here)
  library(argparse)
})

# Parse args and rerun preprocessing.
parser <- ArgumentParser()
parser$add_argument("--config_path")
parser$add_argument("--results_path")
parser$add_argument("--modalities")
parser$add_argument("--data")
parser$add_argument("--n_noise_modalities")
parser$add_argument("--target")
parser$add_argument("--n_noise_dimensionality")
parser$add_argument("--n_pca_dimensionality")
parser$add_argument("--pca_separate")

args <- parser$parse_args()

config <- rjson::fromJSON(
  file = here::here(args$config_path, "config.json")
)

run_benchmark(
  config_path = args$config_path,
  results_path = args$results_path,
  modalities = config$modality_order[as.integer(strsplit(args$modalities, ",")[[1]]) + 1],
  input = args$data,
  n_noise_modalities = as.numeric(args$n_noise_modalities),
  target = as.logical(args$target == "true"),
  n_noise_dimensionality = as.numeric(args$n_noise_dimensionality),
  n_pca_dimensionality = as.numeric(args$n_pca_dimensionality),
  pca_separate = args$pca_separate
)