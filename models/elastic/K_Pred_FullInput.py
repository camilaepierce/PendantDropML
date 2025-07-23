"""
NN Model, focusing on rz coordinates for elastic drops, large for accuracy

Last modified: 7.2.2025
"""
import torch
from torch import nn


class K_Modulus_Full(nn.Module):
    """ Works with rc coordinates 40x2, and output features 40x2
    
    Optimal Learning Rate: 0.1
    """

    def __init__(self):
        super().__init__()

        self.k_linear_relu_stack = nn.Sequential(
            # nn.LayerNorm((80, 2)),
            nn.Flatten(),
            # nn.Dropout(p=0.05),
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # self.first = FirstModel
        self.name = "K Prediction"

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        # x = x.unsqueeze(1)
        # drop = nn.Dropout(p=0.1)
        # x = drop(x)
        logits = self.k_linear_relu_stack(x)
        return logits
    
