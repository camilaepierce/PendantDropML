"""
MODEL TEMPLATE

Use for elastic interfaces to predict stress tensor of a 40-sampled image.

To modify model, change the amount and type of layers within self.linear_relu_stack
(Does not need to only be a linear and relu stack)

Last modified: 7.28.2025
"""
import torch
from torch import nn


class NewModel(nn.Module):
    """ Works with rc coordinates 40x2, and output features 40x2"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # Input begins with 80
            nn.Linear(80, 939), # Change output shape and add more layers
            nn.ReLU(),
            nn.Linear(939, 2), # Change output shape and add more layers
            nn.ReLU(),
            nn.Linear(2, 80), # Change output shape and add more layers
            nn.ReLU(),
            # Repeat, can also change Linear (summation) or ReLU (activation)
            # Output ends with 80
        )
        self.name = "NewModel"

    def forward(self, x):
        # OPTIONAL: normalization layer
        # x = torch.nn.functional.normalize(x)
        # OPTIONAL: dropout layer
        # d = nn.Dropout(p=0.1)
        # x = d(x)
        x = torch.flatten(x, start_dim=1) # Arrange 40 x 2 input into array of size 80
        logits = self.linear_relu_stack(x) # Run input through stack
        logits = torch.unflatten(logits, 1, (40, 2)) # Rearrange array of size 80 into 40 x 2 output

        return logits
    
