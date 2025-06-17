import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch


def conv3d_example():
    # for deterministic output only
    torch.random.manual_seed(0)
    N,C,D,H,W = 126, 3, 1, 656, 875
    img = torch.randn(N,C,H,W)
    img = img.unsqueeze(2)
    ##
    in_channels = C
    out_channels = 1
    kernel_size = (1, 3, 3)
    conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
    ##
    out = conv(img)
    print(out)
    print(out.size())


conv3d_example()