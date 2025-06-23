import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.dataloader import PendantDataLoader
from utils.extraction import PendantDropDataset
from skimage.color import rgb2gray
from torchvision import transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class GrayscaleTransform(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
	    nn.LayerNorm([656, 875]),
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 3, 3)),
            # torch.squeeze(),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=1, kernel_size=(1, 9, 9)),
            nn.MaxPool3d(kernel_size=(1, 15, 15), stride=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(59924, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.Linear(300, 1)
        )
        self.name = "Five Layer CNN"

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = transforms.Grayscale()(x)
        x = x.unsqueeze(2)
        # print("Forward x shape", x.shape)
        logits = self.linear_relu_stack(x)
        logits = logits.squeeze()
        return logits
    
