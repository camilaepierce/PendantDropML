"""
NN Model, focusing on rz coordinates, series of linear and ReLU activations.

Last modified: 6.26.2025
"""
import torch
from torch import nn


class Elastic(nn.Module):
    """ Works with rc coordinates 40x2, and output features 40x2"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(80, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 80)
        )
        self.name = "Five Layer CNN"

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        # x = x.squeeze()
        x = torch.flatten(x, start_dim=1)
        logits = self.linear_relu_stack(x)
        logits = torch.unflatten(logits, 1, (40, 2))
        logits = logits.squeeze()
        return logits
    
