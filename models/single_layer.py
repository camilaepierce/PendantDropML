from torch import nn

class SingleLayerCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_size = (3, 875, 656)
        self.linear_line_size = int(16*(self.image_size[1]//4)*(self.image_size[2]//4))
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(in_features=163, out_features=128),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):
        # x = x.squeeze()
        # print("Forward x shape", x.shape)
        # x = self.flatten(x)
        x = x.permute(0, 3, 2, 1)
        logits = self.linear_relu_stack(x)
        return logits