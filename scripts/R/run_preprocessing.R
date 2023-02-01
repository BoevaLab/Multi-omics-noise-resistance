#' Helper function to rerun our complete preprocessing in R.
#'
#' @returns NULL. All preprocessed datasets are directly written to disk.
rerun_preprocessing_R <- function() {
  suppressPackageStartupMessages({
    library(here)
    library(rjson)
    library(tibble)
    library(maftools)
    library(vroom)
    source(here::here("noise_resistance", "R", "prep", "prepare_tcga_data.R"))
    source(here::here("noise_resistance", "R", "prep", "read_tcga_data.R"))
  })

  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  # Increase VROOM connection size for larger PANCAN files.
  Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 8)

  # Read in all PANCAN files.
  gex_master <- vroom(
    here::here(
      "data", "raw", "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
    )
  ) %>% data.frame(check.names = FALSE)

  cnv_master <- vroom(here::here(
    "data", "raw", "TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
  )) %>% data.frame(check.names = FALSE)

  meth_master <- vroom(here::here(
    "data", "raw", "jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv"
  )) %>% data.frame(check.names = FALSE)

  mirna_master <- vroom(here::here(
    "data", "raw", "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv"
  )) %>%
    data.frame(check.names = FALSE)

  mutation <- maftools::read.maf(here::here(
    "data", "raw",
    "mc3.v0.2.8.PUBLIC.maf.gz"
  ))

rppa_master <- vroom::vroom(here::here(
    "data", "raw", "TCGA-RPPA-pancan-clean.txt"
  )) %>% data.frame(check.names = FALSE)

  # Create patient by gene matrix for mutation. We count only non-silent mutations
  # (which is the default of `maftools::mutCountMatrix`) and remove non-mutated
  # genes (since they would otherwise anyway be removed by the zero-variance)
  # filter.
  mut_master <- mutCountMatrix(mutation,
    removeNonMutated = TRUE
  )

  clinical <- read_raw_clinical_data()
  tcga_cdr <- clinical$clinical_data_resource_outcome
  tcga_w_followup <- clinical$clinical_with_followup

  for (cancer in config$datasets) {
    prepare_new_cancer_dataset(
      cancer = cancer, include_rppa = TRUE,
      tcga_cdr, tcga_w_followup,
      gex_master, cnv_master,
      meth_master, rppa_master,
      mirna_master, mut_master

    )
  }
  return(NULL)
}

suppressPackageStartupMessages({
  library(here)
})

rerun_preprocessing_R()