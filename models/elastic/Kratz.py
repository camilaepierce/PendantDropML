"""
NN Model, focusing on rz coordinates for elastic drops, large for accuracy

Last modified: 7.2.2025
"""
import torch
from torch import nn


class Extreme(nn.Module):
    """ Works with rc coordinates 40x2, and output features 40x2
    
    Optimal Learning Rate: 0.1
    """

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 80)
        )
        self.name = "Kratz Structured CNN"

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        # x = x.unsqueeze(1)
        # drop = nn.Dropout(p=0.1)
        # x = drop(x)
        logits = self.linear_relu_stack(x)
        logits = torch.unflatten(logits, 1, (40, 2))
        logits = logits.squeeze()
        return logits
    
