import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class FiveLayerCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten == nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model = FiveLayerCNN().to(device)
print(model)