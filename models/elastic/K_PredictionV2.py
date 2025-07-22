"""
NN Model, focusing on rz coordinates for elastic drops, large for accuracy

Last modified: 7.2.2025
"""
import torch
from torch import nn


class K_ModulusV2(nn.Module):
    """ Works with rc coordinates 40x2, and output features 40x2
    
    Optimal Learning Rate: 0.1
    """

    def __init__(self):
        super().__init__()
        # self.tens_linear_relu_stack = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(80, 4000),
        #     nn.ReLU(),
        #     nn.Linear(4000, 80),
        # )
        self.k_linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(1),
        )
        # self.first = FirstModel
        self.name = "K Prediction"

    def forward(self, x):
        # x = torch.nn.functional.normalize(x)
        # x = x.unsqueeze(1)
        # drop = nn.Dropout(p=0.1)
        # x = drop(x)
        # tens_logits = self.tens_linear_relu_stack(x)
        # tens_logits = tens_logits.squeeze()
        # k_logits = self.k_linear_relu_stack(torch.cat((nn.Flatten()(x), tens_logits), dim=1))
        # self.first.eval()
        # self.first.requires_grad_(False)
        # tens = self.first(x)
        # tens.requires_grad_(False)
        # x.requires_grad_(False)
        # print(tens.shape, x.shape)
        # print(torch.flatten(tens, start_dim=1).shape, torch.flatten(x, start_dim=1).shape)
        # x = x.flatten(start_dim=1)
        # tens = tens.flatten(start_dim=1)
        # print(torch.cat((x, tens), dim=1).shape)
        # tens.requires_grad_(False)
        # x.requires_grad_(False)
        # input = torch.cat((x, tens), dim=1)
        # input.requires_grad_(False)
        logits = self.k_linear_relu_stack(x)
        return logits
    
