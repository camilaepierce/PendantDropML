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
# from torchinfo import summary

from utils.optimizer import run_optimizer
from utils.evaluation import evaluate_directory

# from models.simple.image_input.five_layer import FiveLayerCNN
# from models.simple.image_input.grayscaletransform import GrayscaleTransform
# from models.simple.rz_input.Xanathor import Xanathor
# from models.elastic.elasticbasic import Elastic
# from models.elastic.Gandalf import Gandalf
from models.elastic.Empty import Empty


if __name__ == "__main__":

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")


    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    model = Empty()
    # model.load_state_dict(torch.load('model_weights/Gandalf.pth', weights_only=True))

    # Run the optimzer
    model = run_optimizer(config, Empty, model=model)


    # print(str(summary(model, input_size=(100, 40, 2))))
    evaluate_directory(model, config, input_type="coordinates")
    # prediction = evaluate_single(model, "data/test_images/2083.png")
    # print(prediction)

    # prediction_tensor = evaluate_directory(model, )
