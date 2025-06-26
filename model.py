"""
Main file for running and training models.

Last modified: 6.26.2025
"""
from json import load
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
# import torchvision.models as models
import utils
from utils.optimizer import run_optimizer
# from models.five_layer import FiveLayerCNN
# from models.grayscaletransform import GrayscaleTransform
from utils.evaluation import evaluate_directory
from models.Xanathor import Xanathor

if __name__ == "__main__":

    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    model = Xanathor()
    model.load_state_dict(torch.load('model_weights/XanathorFull.pth', weights_only=True))

    # Run the optimzer
    model = run_optimizer(config, Xanathor, model=model)



    # evaluate_directory(model, config)
    # prediction = evaluate_single(model, "data/test_images/2083.png")
    # print(prediction)

    # prediction_tensor = evaluate_directory(model, )
