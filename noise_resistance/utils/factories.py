import torch
from noise_resistance.model.criterion import (
    cox_ae_criterion,
    cox_ph_criterion,
)
from noise_resistance.model.fusion import (
    EarlyFusion,
    IntermediateFusionAE,
    IntermediateFusionAttention,
    IntermediateFusionConcat,
    IntermediateFusionEmbrace,
    IntermediateFusionMax,
    IntermediateFusionMean,
    LateFusionMean,
    LateFusionMoE,
    EarlyFusionAE,
)
from noise_resistance.model.skorch_infra import (
    CoxPHAENeuralNet,
    CoxPHNeuralNet,
)
from skorch.callbacks import EarlyStopping, LRScheduler
from noise_resistance.utils.misc_utils import StratifiedSkorchSurvivalSplit


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
        "optimizer": torch.optim.AdamW,
        "max_epochs": 100,
        "lr": 0.001,
        "batch_size": -1,
        "train_split": StratifiedSkorchSurvivalSplit(10, stratified=True),
        "verbose": False,
        "callbacks": [
            (
                "sched",
                LRScheduler(
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    patience=3,
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
        ],
        "module__activation": torch.nn.ReLU,
        "module__modality_hidden_layer_size": 128,
        "module__modality_hidden_layers": 1,
        "module__modality_dimension": 128,
        "module__p_dropout": 0.0,
    },
    "common_tuned": {},
    "common_reconstruction": {"module__alpha": [0.01, 0.1, 1.0]},
    "ae_tuned": {"module__alpha": [0.01, 0.1, 1.0]},
    "early": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "early_ae": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "late_mean": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "late_moe": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "intermediate_mean": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
    "intermediate_max": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
    "intermediate_concat": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
    "intermediate_ae": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
    "intermediate_embrace": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
    "intermediate_attention": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
    },
}

MODALITY_FACTORY = {
    "0": "clinical",
    "0,1": "clinical_gex",
    "0,2": "clinical_rppa",
    "0,3": "clinical_mirna",
    "0,4": "clinical_mutation",
    "0,5": "clinical_methylation",
    "0,6": "clinical_cnv",
    "0,1,2,3,4,5,6": "all",
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
