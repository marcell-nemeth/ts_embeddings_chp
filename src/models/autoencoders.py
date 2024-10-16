from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange


class AutoEncoder(nn.Module):
    def __init__(self, in_features: int, latent_features: int, hidden_size=20):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=latent_features)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=in_features)
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class LSTMAutoEncoder(nn.Module):
    def __init__(self, in_features: int, latent_features: int):
        super().__init__()

        self.encoder = nn.LSTM(input_size=1, hidden_size=latent_features, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=20),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=in_features)
        )

    def encode(self, x: Tensor) -> Tensor:
        # input shape: (batch, seq, feature)
        if x.ndim == 2:
            x = x.unsqueeze(dim=-1)
        output, _ = self.encoder.forward(x)
        return output[:, -1, :]

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def cosine_similarity(a: Tensor, b: Tensor, eps=1e-10) -> Tensor:
    """
    Compute the cosine similarity between two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
        Tensor: The cosine similarity between the two tensors.
    """
    a_norm = a / (a.norm(dim=1)[:, None] + eps)
    b_norm = b / (b.norm(dim=1)[:, None] + eps)
    return torch.mm(a_norm, torch.transpose(b_norm, 0, 1))


def segment_loss(embeddings: Tensor, segment_ixs: Tensor) -> Tensor:
    similarity = cosine_similarity(embeddings, embeddings)

    loss = torch.tensor(0.0, dtype=torch.float)

    same_segments = segment_ixs == torch.unsqueeze(segment_ixs, dim=-1)

    adjacent_ixs = torch.stack([
        segment_ixs-1,   # Segment before
        segment_ixs+1])  # Segment after
    diff_segments = (segment_ixs == adjacent_ixs.unsqueeze(dim=-1)).sum(dim=0, dtype=torch.bool)

    for sim, same, diff in zip(similarity, same_segments, diff_segments):
        # Windows from the same segment should equal 1.0 in cosine similarity
        num_same = same.sum()
        if 0 < num_same:
            loss += F.mse_loss(sim[same], torch.ones(num_same), reduction="mean")

        # Windows from different segments should equal 0.0 in cosine similarity
        num_diff = diff.sum()
        if 0 < num_diff:
            loss += F.mse_loss(sim[diff], torch.zeros(num_diff), reduction="mean")

    return loss


def train(model: AutoEncoder, segments: List[List[Tensor]], **kwargs) -> None:
    """
    Train the given autoencoder model using the provided segments.

    Args:
    - model (AutoEncoder): The autoencoder model to be trained.
    - segments (List[List[Tensor]]): A list of segments, where each segment is represented as a list of tensors.
    - **kwargs: Additional keyword arguments for training configuration.
        - lr (float): Learning rate for the optimizer. Default is 0.001.
        - batch_size (int): Batch size for training. Default is 64.
        - epochs (int): Number of training epochs. Default is 1000.
        - latent_loss (float): With what weight to include the latent loss, if
            zero is specified then it won't be included, Default is 0.

    This function trains the provided autoencoder model using the specified segments.
    It utilizes mean squared error (MSE) loss for reconstruction and optionally includes
    a latent loss if 'latent_loss' is greater than 0.
    """
    # Extract and set training parameters
    params = dict(
        lr=0.001,
        batch_size=64,
        epochs=1000,
        latent_loss=0.0,
    )
    params.update(**kwargs)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), params['lr'])

    # Define data loader, each record is a window and its corresponding segment's index
    train_loader = DataLoader(
        [(window, i)
         for i, segment_windows in enumerate(segments)
         for window in segment_windows],
        params['batch_size'],
        shuffle=True)

    # Initialize variables to track the best model parameters and loss
    best_params, best_loss = None, 0

    # Training loop
    for _ in (t := trange(1, params['epochs']+1, desc=f'Training {model.__class__.__name__}', unit='epoch')):
        loss_sum = 0
        for inputs, segment_ixs in train_loader:
            optimizer.zero_grad()

            # Forward pass
            embeddings = model.encode(inputs)
            outputs = model.decode(embeddings)

            # Compute reconstruction and latent loss if necessary
            loss = F.mse_loss(input=outputs, target=inputs, reduction='none').mean(dim=1).sum()
            if 0 < params["latent_loss"]:
                loss += params["latent_loss"] * segment_loss(embeddings, segment_ixs)
            assert torch.isnan(loss) == False

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # Compute average loss for the epoch
        avg_loss = loss_sum / len(train_loader)

        # Update best model parameters if necessary
        if best_params is None or avg_loss < best_loss:
            best_params = model.state_dict()
            best_loss = avg_loss

        t.set_postfix({'avg_loss': avg_loss, 'best_loss': best_loss})

    # Load best model parameters after training
    model.load_state_dict(best_params)
