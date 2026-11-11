"""PyTorch MLP classifier."""

import torch
from torch import nn

from src import config


class ActivityClassifier(nn.Module):
    """One hidden layer: Linear -> BatchNorm -> ReLU -> Dropout -> Linear.

    A single hidden layer is enough here: in MotionFuse, deeper networks did
    not beat a one-layer MLP on small per-split training sets.
    """

    def __init__(self, input_dim: int, num_classes: int = config.NUM_CLASSES,
                 hidden_dim: int = config.HIDDEN_DIM, dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of feature vectors."""
        return self.net(x)
