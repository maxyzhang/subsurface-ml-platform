import torch
import torch.nn as nn


class LithologyNeuralNetwork(nn.Module):
    """
    Simple feed-forward neural network for lithology classification.
    """

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)