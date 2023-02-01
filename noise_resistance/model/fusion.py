import torch
from noise_resistance.imports.multisurv_imports import (
    EmbraceNet,
    Attention,
)
from noise_resistance.utils.module_utils import (
    HazardRegression,
    Encoder,
    Decoder,
    AE
)


class Fusion(torch.nn.Module):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        self.modality_dimension = modality_dimension

    def forward(self, x):
        raise NotImplementedError


class EarlyFusion(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )

    def forward(self, x):
        return x


class EarlyFusionAE(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.ae = AE(
            input_dimension=sum([len(block) for block in blocks]),
            hidden_layer_size=log_hazard_hidden_layer_size,
            activation=activation,
            hidden_layers=log_hazard_hidden_layers - 1,
            embedding_dimension=log_hazard_hidden_layer_size,
            p_dropout=p_dropout,
        )

    def forward(self, x):
        return self.ae(x)


class LateFusion(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.modality_specific_log_hazard_ratio = torch.nn.ModuleList(
            [
                HazardRegression(
                    input_dimension=len(blocks[i]),
                    hidden_layer_size=log_hazard_hidden_layer_size,
                    activation=activation,
                    hidden_layers=log_hazard_hidden_layers,
                    p_dropout=p_dropout,
                )
                for i in range(len(blocks))
            ]
        )

    def get_weights(self, x):
        raise NotImplementedError

    def forward(self, x):
        return torch.squeeze(
            torch.stack(
                [
                    self.modality_specific_log_hazard_ratio[i](
                        x[:, self.blocks[i]]
                    )
                    for i in range(len(self.blocks))
                ]
            )
        ).T.mm(self.get_weights(x))


class LateFusionMean(LateFusion):
    def get_weights(self, x):
        return torch.full_like(
            torch.ones((len(self.blocks), 1)), 1 / len(self.blocks)
        )


class LateFusionMoE(LateFusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.modality_specific_log_hazard_ratio = torch.nn.ModuleList(
            [
                HazardRegression(
                    input_dimension=len(blocks[i]),
                    hidden_layer_size=log_hazard_hidden_layer_size,
                    activation=activation,
                    hidden_layers=log_hazard_hidden_layers,
                    p_dropout=p_dropout,
                )
                for i in range(len(blocks))
            ]
        )
        self.weights = torch.nn.Sequential(
            torch.nn.Linear(
                sum([len(block) for block in blocks]),
                len(blocks),
            ),
            torch.nn.Softmax(dim=1),
        )

    def get_weights(self, x):
        return self.weights(x)

    def forward(self, x):
        return torch.unsqueeze(
            torch.sum(
                torch.squeeze(
                    torch.stack(
                        [
                            self.modality_specific_log_hazard_ratio[i](
                                x[:, self.blocks[i]]
                            )
                            for i in range(len(self.blocks))
                        ]
                    )
                ).T
                * self.get_weights(
                    torch.cat(
                        [
                            x[:, self.blocks[i]]
                            for i in range(len(self.blocks))
                        ],
                        axis=1,
                    ),
                ),
                axis=1,
            ),
            1,
        )


class IntermediateFusion(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
        reconstructed=False,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.reconstructed = reconstructed
        self.modality_encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dimension=len(blocks[i]),
                    reconstructed=self.reconstructed,
                    hidden_layer_size=modality_hidden_layer_size,
                    activation=activation,
                    hidden_layers=modality_hidden_layers,
                    embedding_dimension=modality_dimension,
                    p_dropout=p_dropout,
                )
                for i in range(len(blocks))
            ]
        )

    def fusion(self, x):
        raise NotImplementedError

    def forward(self, x):
        modality_encodings = []
        for ix, modality in enumerate(self.blocks):
            modality_encodings.append(
                self.modality_encoders[ix](x[:, modality])
            )

        return self.fusion(modality_encodings)


class IntermediateFusionMean(IntermediateFusion):
    def fusion(self, x):
        return torch.mean(torch.stack(x), axis=0)


class IntermediateFusionMax(IntermediateFusion):
    def fusion(self, x):
        # `torch.max` returns a tuple of (max, max_indices),
        # so we select the maximum as the first element.
        return torch.max(torch.stack(x), axis=0)[0]


class IntermediateFusionConcat(IntermediateFusion):
    def fusion(self, x):
        return torch.concat(x, axis=1)


class IntermediateFusionAE(IntermediateFusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.modality_encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dimension=len(blocks[i]),
                    hidden_layer_size=modality_hidden_layer_size,
                    activation=activation,
                    hidden_layers=modality_hidden_layers,
                    embedding_dimension=modality_dimension,
                    p_dropout=p_dropout,
                    reconstructed=True,
                )
                for i in range(len(blocks))
            ]
        )
        self.latent_space = torch.nn.Linear(
            modality_dimension * len(blocks), modality_dimension * len(blocks)
        )
        self.joint_decoded = torch.nn.Linear(
            modality_dimension * len(blocks), modality_dimension * len(blocks)
        )
        self.modality_decoders = torch.nn.ModuleList(
            Decoder(
                encoder=Encoder(
                    input_dimension=len(blocks[ix]),
                    hidden_layer_size=modality_hidden_layer_size,
                    activation=activation,
                    hidden_layers=modality_hidden_layers,
                    embedding_dimension=modality_dimension,
                    p_dropout=p_dropout,
                    reconstructed=True,
                ),
                activation=activation,
                p_dropout=p_dropout,
                output_dimensionality=len(blocks[ix]),
            )
            for ix in range(len(self.modality_encoders))
        )

    def fusion(self, x):
        latent_space = self.latent_space(torch.concat(x, axis=1))
        decoded_latent_space = self.joint_decoded(latent_space)
        decoded = torch.concat(
            [
                self.modality_decoders[ix](
                    decoded_latent_space[
                        :,
                        ix
                        * self.modality_dimension : (ix + 1)
                        * self.modality_dimension,
                    ]
                )
                for ix in range(len(self.modality_decoders))
            ],
            axis=1,
        )
        return (latent_space, decoded)


class IntermediateFusionEmbrace(IntermediateFusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.embrace = EmbraceNet()

    def fusion(self, x):
        return self.embrace(torch.stack(x))


class IntermediateFusionAttention(IntermediateFusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        modality_dimension,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
            modality_dimension,
        )
        self.attend = Attention(size=modality_dimension)

    def fusion(self, x):
        return self.attend(torch.stack(x))
