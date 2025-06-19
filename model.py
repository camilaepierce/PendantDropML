from json import load
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import utils
from utils.optimizer import run_optimizer
# from models.five_layer import FiveLayerCNN
from models.grayscaletransform import GrayscaleTransform


if __name__ == "__main__":

    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    # Run the optimzer
    model = run_optimizer(config, GrayscaleTransform)

    with open(config["save_info"]["modelName"] + ".txt", "a") as f:
        f.write(str(config))
        f.write([print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n") for name, param in model.named_parameters()])