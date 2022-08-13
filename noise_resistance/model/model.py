import torch
from noise_resistance.utils.factories import FUSION_FACTORY
from noise_resistance.utils.misc_utils import calculate_log_hazard_input_size
from noise_resistance.utils.module_utils import (
    HazardRegression,
)


class CoxPHNeural(torch.nn.Module):
    def __init__(
        self,
        fusion_method,
        blocks,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        modality_dimension=64,
        alpha=0.01,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        self.modality_dimension = modality_dimension
        self.alpha = alpha
        self.fusion = FUSION_FACTORY[fusion_method](
            blocks=blocks,
            activation=activation,
            p_dropout=p_dropout,
            log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
            log_hazard_hidden_layers=log_hazard_hidden_layers,
            modality_hidden_layer_size=modality_hidden_layer_size,
            modality_hidden_layers=modality_hidden_layers,
            modality_dimension=modality_dimension,
        )
        if "late" not in fusion_method:
            self.log_hazard = HazardRegression(
                input_dimension=calculate_log_hazard_input_size(
                    fusion_method=fusion_method,
                    blocks=blocks,
                    modality_dimension=modality_dimension,
                ),
                hidden_layer_size=log_hazard_hidden_layer_size
                // (1 + int(fusion_method == "early_ae")),
                activation=activation,
                hidden_layers=log_hazard_hidden_layers
                - int(fusion_method == "early_ae"),
                p_dropout=p_dropout,
            )

    def forward(self, x):
        fused = self.fusion(x)
        if "late" in self.fusion_method:
            return fused
        else:
            if len(fused) > 1 and isinstance(fused, tuple):
                if self.fusion_method == "intermediate_ae":
                    return (self.log_hazard(fused[0]),) + fused[1:] + (x,)
                else:
                    return (self.log_hazard(fused[0]),) + fused[1:]
            else:
                return self.log_hazard(fused)
