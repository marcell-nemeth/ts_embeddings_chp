from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super(ResidualBlock, self).__init__()

        full_kernel_width = (kernel_size - 1) * dilation + 1
        self.left_padding = full_kernel_width - 1

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.upsample = None
        if in_channels != out_channels:
            self.upsample = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )

    def forward(self, x: Tensor):
        out = self.conv1(F.pad(x, (self.left_padding, 0)))  #   causal convolution
        out = self.norm1(out)  # batch normalization
        out = F.relu(out)
        out = self.conv2(F.pad(out, (self.left_padding, 0)))  # causal convolution
        out = self.norm2(out)  # batch normalization
        out = out + (x if self.upsample is None else self.upsample(x))
        return F.relu(out)


class Decoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
            # NOTE: Not really needed since we don't have dynamic covariates
            # nn.BatchNorm1d(out_features),
            # nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DeepTCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        latent_features: int,
        kernel_size: int = 2,
        dilation_rates: List[int] = [1, 2, 4, 8],
        hidden_size=20,
    ):
        super(DeepTCN, self).__init__()

        self.in_features = in_features
        self.latent_features = latent_features

        # Encoder input shape: [batch, 1, window_size]
        # Encoder output shape: [batch, latent_features, window_size]
        self.encoder = nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=1 if i == 0 else latent_features,
                    out_channels=latent_features,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i, dilation in enumerate(dilation_rates)
            )
        )

        # NOTE: Can't make the window size the embdding dimension because then the
        #       convolutions's weights would be just: [0, 0, ..., 1]
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Decoder input shape: [batch, channels]
        # Decoder output shape: [batch, window_size]
        self.decoder = Decoder(
            in_features=latent_features, out_features=in_features, hidden_size=hidden_size
        )

    def encode(self, x: Tensor):
        x = x.reshape((-1, 1, self.in_features))
        x = self.encoder(x)
        x = torch.reshape(self.max_pool(x), (-1, self.latent_features))
        return x

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


if __name__ == "__main__":
    # Test run
    model = DeepTCN(in_features=25, latent_features=8)
    print(model)

    x = torch.rand((32, 25))
    print(f"Input shape : {x.shape}")
    embedding = model.encode(x)
    print(f"Latent shape: {embedding.shape}")
    output = model.decode(embedding)
    print(f"Output shape: {output.shape}")
