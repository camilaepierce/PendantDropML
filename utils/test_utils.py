# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import io
import torch
# import random

example = torch.rand((3, 5))

numpy_version = example.detach().numpy()

print(numpy_version)
print(type(numpy_version))
print(f"mean: {torch.mean(example)} std: {torch.std(example)}")