suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(rjson)
  library(fastDummies)
  library(tidyr)
  library(janitor)
  library(forcats)
  library(stringr)
})

#' Chooses the proper sample in case donors have multiple primary samples.
#' We first chose the "lower" vial (i.e., we pick vial A over vial B, since
#' the lower vials tend to be much more common). Afterward, if there are still
#' multiple samples for a donor, we follow the official guidance from the broad
#' institute by choosing the lexicographically smaller barcode.
#'
#' @param barcodes vector. Vector containing two or more barcodes
#'                         for the same patient.
#'
#' @returns character. We return only one "chosen" barcode, based on the logic above.
choose_proper_sample <- function(barcodes) {
  vials <- substr(barcodes, 16, 16)
  if (length(unique(vials)) > 1) {
    return(str_sort(barcodes, numeric = TRUE)[1])
  } else {
    return(str_sort(barcodes, numeric = TRUE, decreasing = TRUE)[1])
  }
}

#' Filters TCGA samples based on their barcode. Excludes everything but non-primary
#' samples. Also makes sure that each donor is represented
#' through only one sample (see `choose_proper_sample`).
#'
#'
#' @param df data.frame. data.frame containing the data in question. Colnames must
#'                       be TCGA barcodes, rows are expected to contain molecular features
#'                       (although we don't touch them here).
#'
#' @returns data.frame. data.frame filtered based on the principles above.
#'                      This new data.frame will contain only samples of the
#'                      specified types and additionally contain only one
#'                      sample for each donor.
filter_samples <- function(df) {
  # Get tissue types from barcodes.
  types <- substr(colnames(df), 14, 15)

  # 01 - Primary Solid Tumor
  # 03 - Primary Blood Derived Cancer - Peripheral Blood
  # 09 - Primary Blood Derived Cancer - Bone Marrow

  # 04 - Recurrent Blood Derived Cancer - Bone Marrow	TRBM
  # 06 - Metastatic	TM

  # Select only primary tumors.
  selected_types <- c("01", "03", "09")
  df <- df[, types %in% selected_types]
  # The first 12 characters of the barcode uniquely determine the donor
  # in question.
  donors <- substr(colnames(df), 1, 12)

  # Make sure no donors are duplicated by using the logic detailed in
  # `choose_proper_sample`.
  duplicated_donors <- names(which(table(donors) > 1))
  if (length(duplicated_donors) > 0) {
    unique_df <- df[, !donors %in% duplicated_donors]
    df <- cbind(unique_df, df[, unname(sapply(
      unlist(lapply(duplicated_donors, function(x) choose_proper_sample(grep(x, colnames(df), value = TRUE)))),
      function(x) grep(x, colnames(df))
    ))])
  }
  # Cut barcode to only the patient id for modeling.
  colnames(df) <- unname(sapply(colnames(df), function(x) substr(x, 1, 12)))
  return(df)
}

#' Impute data. We use the same imputation logic for all molecular features.
#' Clinical features are handled separately (see our paper for details).
#'
#' @param df data.frame. data.frame containing some missing values that are to be imputed.
#'
#' @returns data.frame. Complete data.frame for which all missing values have been
#'                      either imputed or the covariates in question have been removed.
impute <- function(df) {
  # Columns contain patients - we exclude patients which are missing for more
  # than 10% of all patients.
  large_missing <- which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10))
  if (length(large_missing) > 0) {
    df <- df[-which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10)), ]
  }

  # Any missing values still left over after this initial filtering step are
  # imputed using the median value per feature.
  if (any(is.na(df))) {
    for (na_count in which(apply(df, 1, function(x) any(is.na(x))))) {
      df[na_count, ] <- as.numeric(df[na_count, ]) %>% replace_na(median(as.numeric(df[na_count, ]), na.rm = TRUE))
    }
  }
  return(df)
}

#' Performs complete preprocessing of molecular data:
#' - Filtering: Removing non-primary/normal samples and making sure
#'              each donor is represented by only one sample
#' - Imputation: Imputing or removing covariates with missing values
#' - Logging: Some molecular data is best represented in log-space. Thus,
#'            these data are logged here.
#'
#' @param df data.frame. data.frame containing molecular data to be preprocessed.
#' @param log logical. Whether the molecular data should be logged.
#' @returns data.frame. Preprocessed data.frame.
preprocess <- function(df, log = FALSE) {
  if (all(nchar(colnames(df)) >= 15)) {
    df <- filter_samples(df)
  }
  if (any(is.na(df))) {
    df <- impute(df)
  }
  if (log) {
    df <- log(1 + df, base = 2)
  }
  return(df)
}

#' Performs complete preprocessing of TCGA-PANCANATLAS gene expression data.
#'
#' @param gex data.frame. data.frame containing mRNA data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_gene_expression_pancan <- function(gex) {
  rownames(gex) <- gex[, 1]
  gex <- gex[, 2:ncol(gex)]
  gex <- preprocess(gex, log = TRUE)
  return(gex)
}

#' Performs complete preprocessing of TCGA clinical data.
#'
#' @param clinical_raw data.frame. data.frame containing TCGA-CDR clinical data.
#' @param clinical_ext_raw data.frame. data.frame containing TCGA-TSV (clinical with follow-up) clinical data.
#' @param cancer character data.frame. Cancer dataset name to be used.
#' @returns data.frame. Preprocessed data.frame with clinical data.
prepare_clinical_data <- function(clinical_raw, clinical_ext_raw, cancer) {
  # Keep patients that are missing survival information if the user desires.
  clinical <- clinical_raw %>%
      # remove any patients for which the OS endpoint is missing
      filter(!(is.na(OS) | is.na(OS.time))) %>%
      # remove any patients which were not at risk at the start of the study
      filter(!(OS.time == 0))
  # Select out clinical covariates in question and recode
  # all missing data to `NA`.
  clinical <- clinical %>%
    filter(type == cancer) %>%
    dplyr::select(
      bcr_patient_barcode, OS, OS.time,
      age_at_initial_pathologic_diagnosis,
      gender,
      race,
      ajcc_pathologic_tumor_stage,
      clinical_stage,
      histological_type
    ) %>%
    mutate(race = recode(race, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Not Evaluated]` = "NA")) %>%
    mutate(ajcc_pathologic_tumor_stage = recode(ajcc_pathologic_tumor_stage, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(clinical_stage = recode(clinical_stage, `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(histological_type = recode(histological_type, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(race = replace_na(race, "NA")) %>%
    mutate(ajcc_pathologic_tumor_stage = replace_na(ajcc_pathologic_tumor_stage, "NA")) %>%
    mutate(clinical_stage = replace_na(clinical_stage, "NA")) %>%
    mutate(histological_type = replace_na(histological_type, "NA"))
  admin <- clinical[, c("bcr_patient_barcode", "OS", "OS.time")]
  colnames(admin)[1] <- "patient_id"
  # Perform imputation separately.
  # Features with missing numerical values are removed since imputing e.g.,
  # age is not reliable using simple strategies.
  # Categorical variables with missing values are "imputed", where we assign
  # missing values their own `NA` category, since we assume that missingness
  # is not at random.
  numerical <- clinical[, names(which(sapply(clinical[, -(1:3)], is.numeric))), drop = FALSE]
  na_numerical_mask <- apply(numerical, 2, function(x) any(is.na(x)))
  categorical <- clinical[, names(which(!sapply(clinical[, -(1:3)], is.numeric)))]
  clinical <- cbind(admin, numerical[, !na_numerical_mask], categorical)
  return(clinical)
}

#' Performs complete preprocessing of TCGA-PANCANATLAS CNV data.
#'
#' @param cnv data.frame. data.frame containing CNV data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_cnv <- function(cnv) {
  rownames(cnv) <- cnv[, 1]
  cnv <- cnv[, 2:ncol(cnv)]
  cnv <- preprocess(cnv, log = FALSE)
  return(cnv)
}

#' Performs complete preprocessing of TCGA-PANCANATLAS DNA methylation data.
#'
#' @param meth data.frame. data.frame containing DNA methylation data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_meth_pancan <- function(meth) {
  rownames(meth) <- meth[, 1]
  meth <- meth[, 2:ncol(meth)]
  meth <- preprocess(meth, log = FALSE)
  return(meth)
}

#' Performs complete preprocessing of TCGA-PANCANATLAS mutation data.
#'
#' @param mut data.frame. data.frame containing mutation data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_mutation <- function(mut) {
  mut <- preprocess(mut, log = FALSE)
  mut
}

#' Performs complete preprocessing of TCGA-PANCANATLAS protein expression data.
#'
#' @param rppa data.frame. data.frame containing protein expression data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_rppa_pancan <- function(rppa) {
  rownames(rppa) <- rppa[, 1]
  rppa <- t(rppa[, 2:ncol(rppa)])
  rppa <- preprocess(rppa, log = FALSE)
  rppa
}

#' Performs complete preprocessing of TCGA-PANCANATLAS miRNA data.
#'
#' @param rppa data.frame. data.frame containing miRNA data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_mirna_pancan <- function(mirna) {
  rownames(mirna) <- mirna[, 1]
  mirna <- mirna[, 2:ncol(mirna)]
  mrina <- preprocess(mirna, log = TRUE)
}

#' Helper function to perform complete preprocessing for TCGA datasets. Writes
#' datasets directly to disk, separated by complete and missing modality samples.
#'
#' @param cancer character. Cancer dataset to be prepared.
#' @param include_rppa logical. Whether the dataset should contain RPPA data.
#' @param include_mirna logical. Whether the dataset should contain miRNA data.
#' @param include_mutation logical. Whether the dataset should contain mutation data.
#' @param include_methylation logical. Whether the dataset should contain DNA methylation data.
#' @param include_gex logical. Whether the dataset should contain mRNA data.
#' @param include_cnv logical. Whether the dataset should contain CNV data.
#' @returns NULL.
prepare_new_cancer_dataset <- function(cancer,
                                       tcga_cdr_master,
                                       tcga_w_followup_master,
                                       gex_master,
                                       cnv_master,
                                       meth_master,
                                       rppa_master,
                                       mirna_master,
                                       mut_master,
                                       include_rppa = FALSE,
                                       include_mirna = TRUE,
                                       include_mutation = TRUE,
                                       include_methylation = TRUE,
                                       include_gex = TRUE,
                                       include_cnv = TRUE) {
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  # Preprocess modalities one after the other taking input parameters into account.
  # NOTE: We separate complete and missing data modalities by collecting the barcodes
  # per each modality before we append the missing modality samples.
  # NB: Appending the missing modality samples is necessary such that we still
  # have the same features/modalities for all samples in the missing
  # modality samples (even if some modalities are completely absent for some samples).
  clinical <- prepare_clinical_data(tcga_cdr_master, tcga_w_followup_master, cancer = cancer)
  sample_barcodes <- list(clinical$patient_id)
  if (include_gex) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(gex_master)))))
    gex_filtered <- gex_master[, c(1, patients)]
    gex <- prepare_gene_expression_pancan(gex_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(gex)))
  }

  if (include_cnv) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(cnv_master)))))
    cnv_filtered <- cnv_master[, c(1, patients)]
    cnv <- prepare_cnv(cnv_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(cnv)))
  }
  if (include_methylation) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(meth_master)))))
    meth_filtered <- meth_master[, c(1, patients)]
    meth <- prepare_meth_pancan(data.frame(meth_filtered, check.names = FALSE))
    sample_barcodes <- append(sample_barcodes, list(colnames(meth)))
  }
  if (include_rppa) {
    patients <- unlist(sapply(clinical$patient_id, function(x) grep(x, rppa_master$SampleID)))
    if (length(patients) > 0) {
      rppa_filtered <- rppa_master[patients, -c(2)]
      rppa <- prepare_rppa_pancan(data.frame(rppa_filtered, check.names = FALSE))
      sample_barcodes <- append(sample_barcodes, list(colnames(rppa)))
    } else {
      include_rppa <- FALSE
    }
  }

  if (include_mirna) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(mirna_master)))))
    if (length(patients) > 0) {
      mirna_filtered <- mirna_master[, c(1, patients)]
      mirna <- prepare_mirna_pancan(data.frame(mirna_filtered, check.names = FALSE))
      sample_barcodes <- append(sample_barcodes, list(colnames(mirna)))
    } else {
      include_mirna <- FALSE
    }
  }
  if (include_mutation) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(mut_master)))))
    mut_filtered <- mut_master[, patients]
    mutation <- prepare_mutation(mut_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(mutation)))
  }

  # Get set of common (complete) samples and write it to disk.
  common_samples <- Reduce(intersect, sample_barcodes)
  data <- clinical %>%
    filter(patient_id %in% common_samples) %>%
    arrange(desc(patient_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS.time, patient_id))
  if (include_cnv) {
    data <- data %>%
      cbind(
        data.frame(t(cnv), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("cnv_", x))
      )
  }

  if (include_gex) {
    data <- data %>%
      cbind(
        data.frame(t(gex), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("gex_", x))
      )
  }


  if (include_methylation) {
    data <- data %>%
      cbind(
        data.frame(t(meth), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("meth_", x))
      )
  }

  if (include_mirna) {
    data <- data %>%
      cbind(
        data.frame(t(mirna), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mirna_", x))
      )
  }

  if (include_mutation) {
    data <- data %>%
      cbind(
        data.frame(t(mutation), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mutation_", x))
      )
  }

  if (include_rppa) {
    data <- data %>%
      cbind(
        data.frame(t(rppa), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("rppa_", x))
      )
  }
  print(paste0("Writing: ", cancer))
  data %>%
    # Rename to `OS_days` for consistency with other projects/datasets.
    rename(OS_days = OS.time) %>%
    write_csv(
      here::here("data", "processed", "TCGA", paste0(cancer, "_data_preprocessed.csv"))
    )
  return(NULL)
}