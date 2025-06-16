# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
from json import load
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


class PendantNetwork(nn.Module):

    def __init__(self):
        super().__init__()

if __name__ == "main":


    # Defining input structure
    config = "./config.json"
    with open(config, "r") as f:
        config = load(f) # config now json object with configuration details

    