import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        super(ResidualBlock, self).__init__()

        # dense layer with ReLU activation and dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        self.layer_norm = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor):
        x = self.dense(x) + self.skip(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class TiDE(nn.Module):
    def __init__(
        self,
        in_features: int,
        latent_features: int,
        hidden_size: int = 20,
        n_encoders: int = 1,
        n_decoders: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ):
        # NOTE: Will not include the loopback resudual connection since
        #       that would provide a solution for the autoencoder output
        super(TiDE, self).__init__()

        assert n_encoders > 0 and n_decoders > 0

        self.in_features = in_features
        self.latent_features = latent_features

        self.encoder = nn.Sequential(
            *[
                ResidualBlock(
                    in_features if i == 0 else hidden_size,
                    hidden_size,
                    latent_features if i == n_encoders - 1 else hidden_size,
                    dropout,
                    use_layer_norm,
                )
                for i in range(n_encoders)
            ]
        )

        self.decoder = nn.Sequential(
            *[
                ResidualBlock(
                    latent_features if i == 0 else hidden_size,
                    hidden_size,
                    in_features if i == n_decoders - 1 else hidden_size,
                    dropout,
                    use_layer_norm,
                )
                for i in range(n_decoders)
            ]
        )

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


if __name__ == "__main__":
    # Test run
    model = TiDE(in_features=25, latent_features=8)
    print(model)

    x = torch.rand((32, 25))
    print(f"Input shape : {x.shape}")
    embedding = model.encode(x)
    print(f"Latent shape: {embedding.shape}")
    output = model.decode(embedding)
    print(f"Output shape: {output.shape}")
