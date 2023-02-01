suppressPackageStartupMessages({
  library(R6)
  library(mlr3)
  library(mlr3proba)
  library(mlr3tuningspaces)
  library(prioritylasso)
})

# Adapted from: https://github.com/mlr-org/mlr3extralearners/blob/main/R/learner_glmnet_surv_cv_glmnet.R

#' Fits a BlockForest method using `mlr3` and `mlr3proba`.
#' For full documentation of all parameters please refer to the documentation
#' of `prioritylasso`.
LearnerSurvCVPrioritylasso <- R6Class("LearnerSurvCVPrioritylasso",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        block1.penalization = p_lgl(default = TRUE, tags = "train"),
        lambda.type = p_fct(c("lambda.min", "lambda.1se"), default = "lambda.min", tags = "train"),
        standardize = p_lgl(default = TRUE, tags = "train"),
        nfolds = p_int(3, 10, default = 5, tags = "train"),
        cvoffset = p_lgl(default = TRUE, tags = "train"),
        cvoffsetnfolds = p_int(3, 10, default = 5),
        alpha = p_dbl(0, 1, default = 0.95)
      )

      super$initialize(
        id = "surv.cv_prioritylasso",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("distr", "crank", "lp")
      )
    }
  ),
  private = list(
    .train = function(task) {
      suppressPackageStartupMessages({
        source(here::here("noise_resistance", "R", "utils", "utils.R"))
        source(here::here("noise_resistance", "R", "utils", "imports.R"))
        library(prioritylasso)
        library(mlr3misc)
      })

      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      pv$family <- "cox"
      # Determine the priority order - see `get_prioritylasso_block_order`
      # for further details.
      block_order <- get_prioritylasso_block_order(
        target, data, 
        unique(sapply(strsplit(task$feature_names, "\\_"), function(x) x[1])), 
        pv$lambda.type
      )
      # Get feature -> modality index mapping.
      blocks <- get_block_assignment(block_order, task$feature_names)

      # Fit prioritylasso with the specified parameters.
      prioritylasso_fit <- mlr3misc::invoke(
        prioritylasso,
        X = data, Y = target, .args = pv,
        blocks = blocks,
        type.measure = "deviance"
      )

      # Transform prioritylasso coefficient estimates to a `coxph` class
      # for easy survival prediction.
      # NB: The below looks convoluted but it is necessary since
      # prioritylasso does not maintain the same order of covariates as in
      # the original data.frame - thus, we reorder the original
      # data.frame to match the feature order of prioritylasso.
      cox_helper <- transform_cox_model(
        prioritylasso_fit$coefficients[which(prioritylasso_fit$coefficients != 0)],
        data[
          ,
          sapply(
            names(prioritylasso_fit$coefficients[which(prioritylasso_fit$coefficients != 0)]),
            function(x) which(colnames(data) == x)
          )
        ], target
      )
      return(cox_helper)
    },
    .predict = function(task) {
      suppressPackageStartupMessages({
        source(here::here("noise_resistance", "R", "utils", "utils.R"))
        source(here::here("noise_resistance", "R", "utils", "imports.R"))
        library(pec)
        library(mlr3proba)
      })

      newdata <- as_numeric_matrix(ordered_features(task, self))
      newdata <- data.frame(newdata)[, colnames(newdata) %in% names(self$model$coefficients)]
      surv <- pec::predictSurvProb(self$model, newdata, self$model$y[, 1])
      lp <- predict(self$model, newdata)
      return(mlr3proba::.surv_return(
        times = self$model$y[, 1],
        surv = surv,
        lp = lp
      ))
    }
  )
)

mlr_learners$add("surv.cv_prioritylasso", LearnerSurvCVPrioritylasso)
