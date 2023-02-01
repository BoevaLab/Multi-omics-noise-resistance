import torch
from noise_resistance.model.criterion import cox_ae_criterion, cox_ph_criterion
from noise_resistance.model.fusion import (
    EarlyFusion,
    EarlyFusionAE,
    IntermediateFusionAE,
    IntermediateFusionAttention,
    IntermediateFusionConcat,
    IntermediateFusionEmbrace,
    IntermediateFusionMax,
    IntermediateFusionMean,
    LateFusionMean,
    LateFusionMoE,
)
from noise_resistance.model.skorch_infra import (
    CoxPHAENeuralNet,
    CoxPHNeuralNet,
    FixSeed,
)
from noise_resistance.utils.misc_utils import (
    StratifiedSkorchSurvivalSplit,
    negative_partial_log_likelihood_loss,
    SurvivalEpochScoring,
)
from scipy.stats import uniform
from sklearn.metrics import make_scorer
from sklearn.utils.fixes import loguniform
from skorch.callbacks import EarlyStopping, LRScheduler

FUSION_FACTORY = {
    "early": EarlyFusion,
    "early_ae": EarlyFusionAE,
    "late_mean": LateFusionMean,
    "late_moe": LateFusionMoE,
    "intermediate_mean": IntermediateFusionMean,
    "intermediate_max": IntermediateFusionMax,
    "intermediate_concat": IntermediateFusionConcat,
    "intermediate_ae": IntermediateFusionAE,
    "intermediate_embrace": IntermediateFusionEmbrace,
    "intermediate_attention": IntermediateFusionAttention,
}

CRITERION_FACTORY = {
    "early": cox_ph_criterion,
    "early_ae": cox_ae_criterion,
    "late_mean": cox_ph_criterion,
    "late_moe": cox_ph_criterion,
    "intermediate_mean": cox_ph_criterion,
    "intermediate_max": cox_ph_criterion,
    "intermediate_concat": cox_ph_criterion,
    "intermediate_ae": cox_ae_criterion,
    "intermediate_embrace": cox_ph_criterion,
    "intermediate_attention": cox_ph_criterion,
}

HYPERPARAM_FACTORY = {
    "common_fixed": {
        "optimizer": torch.optim.Adam,
        "max_epochs": 100,
        "batch_size": -1,
        "lr": 0.01,
        "train_split": StratifiedSkorchSurvivalSplit(10, stratified=True),
        "verbose": False,
        "callbacks": [
            (
                "sched",
                LRScheduler(
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    patience=5,
                ),
            ),
            (
                "es",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=10,
                    load_best=True,
                ),
            ),
            ("seed", FixSeed(seed=42)),
        ],
        "module__activation": torch.nn.ReLU,
        "module__modality_hidden_layer_size": 128,
        "module__modality_hidden_layers": 1,
        "module__modality_dimension": 64,
    },
    "common_tuned": {
        "optimizer__weight_decay": [
            0.1,
            0.01,
            0.001,
        ],
        "module__p_dropout": [0.0, 0.25, 0.5],
    },
    "multimodal_dropout": {
        "module__p_multimodal_dropout": [0.0, 0.25, 0.5]
        },
    "early_tuned": {},
    "late_mean_tuned": {},
    "late_moe_tuned": {},
    "intermediate_mean_tuned": {},
    "intermediate_max_tuned": {},
    "intermediate_concat_tuned": {},
    "intermediate_embrace_tuned": {},
    "intermediate_attention_tuned": {},
    "early_fixed": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "early_ae_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 2,
    },
    "late_mean_fixed": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "late_moe_fixed": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "intermediate_mean_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "intermediate_max_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "intermediate_concat_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "intermediate_ae_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "intermediate_embrace_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "intermediate_attention_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
}

MODALITY_FACTORY = {
    "0": "clinical",
    "1": "gex",
    "2": "rppa",
    "3": "mirna",
    "4": "mutation",
    "5": "methylation",
    "6": "cnv",
    "0,1": "clinical_gex",
    "0,2": "clinical_rppa",
    "0,3": "clinical_mirna",
    "0,4": "clinical_mutation",
    "0,5": "clinical_methylation",
    "0,6": "clinical_cnv",
    "0,1,2,3,4,5,6": "clinical_gex_rppa_mirna_mutation_meth_cnv",
}

SKORCH_NET_FACTORY = {
    "early": CoxPHNeuralNet,
    "early_ae": CoxPHAENeuralNet,
    "late_mean": CoxPHNeuralNet,
    "late_moe": CoxPHNeuralNet,
    "intermediate_mean": CoxPHNeuralNet,
    "intermediate_max": CoxPHNeuralNet,
    "intermediate_concat": CoxPHNeuralNet,
    "intermediate_ae": CoxPHAENeuralNet,
    "intermediate_embrace": CoxPHNeuralNet,
    "intermediate_attention": CoxPHNeuralNet,
}
