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
from models.grayscaletransform import GrayscaleTransform
from utils.evaluation import evaluate_directory


if __name__ == "__main__":

    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    # # Run the optimzer
    model = run_optimizer(config, GrayscaleTransform)


    # model = GrayscaleTransform()
    # model.load_state_dict(torch.load('model_weights/grayscaleFirstMini.pth', weights_only=True))

    evaluate_directory(model, config)
    # prediction = evaluate_single(model, "data/test_images/2083.png")
    # print(prediction)

    # prediction_tensor = evaluate_directory(model, )
