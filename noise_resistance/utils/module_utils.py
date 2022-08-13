import torch


class HazardRegression(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_size=32,
        activation=torch.nn.ReLU,
        hidden_layers=1,
        p_dropout=0.5,
    ):
        super().__init__()
        hazard = []
        current_size = input_dimension
        for layer in range(hidden_layers):
            next_size = hidden_layer_size // (1 + layer)
            hazard.append(torch.nn.Linear(current_size, next_size))
            hazard.append(activation())
            hazard.append(torch.nn.Dropout(p_dropout))
            hazard.append(torch.nn.BatchNorm1d(next_size))
            current_size = next_size
        hazard.append(torch.nn.Linear(current_size, 1, bias=False))
        self.hazard = torch.nn.Sequential(*hazard)

    def forward(self, x):
        return self.hazard(x)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        reconstructed=False,
        hidden_layer_size=128,
        activation=torch.nn.ReLU,
        hidden_layers=1,
        embedding_dimension=64,
        p_dropout=0.5,
    ):
        super().__init__()
        encoder = []
        current_size = input_dimension
        next_size = hidden_layer_size
        for i in range(hidden_layers):
            if i != 0:
                current_size = next_size
                next_size = max(int(next_size / 2), embedding_dimension)
            encoder.append(torch.nn.Linear(current_size, next_size))
            encoder.append(activation())
            encoder.append(torch.nn.Dropout(p_dropout))
            encoder.append(torch.nn.BatchNorm1d(next_size))
        if reconstructed:
            encoder.append(torch.nn.Linear(next_size, embedding_dimension))
        self.encode = torch.nn.Sequential(*encoder)
        self.input_dimension = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.embedding_dimension = embedding_dimension

    def forward(self, x):
        return self.encode(x)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        encoder,
        output_dimensionality,
        activation=torch.nn.ReLU,
        p_dropout=0.5,
    ):
        super().__init__()
        decoder = []
        for layer in encoder.encode[::-1][::4][:-1]:
            current_size = layer.weight.shape[0]
            next_size = layer.weight.shape[1]
            decoder.append(torch.nn.Linear(current_size, next_size))
            decoder.append(activation())
            decoder.append(torch.nn.Dropout(p_dropout))
            decoder.append(torch.nn.BatchNorm1d(next_size))
        decoder.append(torch.nn.Linear(next_size, output_dimensionality))
        self.decode = torch.nn.Sequential(*(decoder))

    def forward(self, x):
        return self.decode(x)


class AE(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_size=128,
        activation=torch.nn.ReLU,
        hidden_layers=2,
        embedding_dimension=64,
        p_dropout=0.5,
    ):
        super().__init__()
        self.encode = Encoder(
            input_dimension=input_dimension,
            reconstructed=True,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            hidden_layers=hidden_layers,
            embedding_dimension=embedding_dimension,
            p_dropout=p_dropout,
        )
        self.decode = Decoder(
            encoder=self.encode,
            activation=activation,
            p_dropout=p_dropout,
            output_dimensionality=input_dimension,
        )
        self.input_dimension = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.embedding_dimension = embedding_dimension

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded, x
