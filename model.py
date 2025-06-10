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


if __name__ == "main":


    # Defining input structure
    config = "./config.json"
    with open(config, "r") as f:
        config = load(f) # config now json object with configuration details
    model.add(keras.Input(shape=(config.input.IMG_HEIGHT, config.input.IMG_WIDTH))) # HEIGHT x WIDTH black and white images as input
    model.add(layers.)