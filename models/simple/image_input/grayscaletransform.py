import torch
from torch import nn
from torchvision import transforms
from torchvision import transforms

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
    
