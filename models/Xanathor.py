"""
NN Model, focusing on rz coordinates, series of linear and ReLU activations.

Last modified: 6.26.2025
"""
import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class Xanathor(nn.Module):
    """ Works with rc coordinates 40x2"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.name = "Five Layer CNN"

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        # x = x.squeeze()
        logits = self.linear_relu_stack(x)
        logits = logits.squeeze()
        return logits
    
