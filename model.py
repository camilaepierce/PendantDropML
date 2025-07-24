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

from utils.k_optimizer import run_optimizer
from utils.evaluation import evaluate_directory

# from models.simple.image_input.five_layer import FiveLayerCNN
# from models.simple.image_input.grayscaletransform import GrayscaleTransform
# from models.simple.rz_input.Xanathor import Xanathor
# from models.elastic.elasticbasic import Elastic
# from models.elastic.Gandalf import Gandalf
# from models.elastic.Empty import Empty
# from models.elastic.Extreme2 import Extreme
# from models.elastic.K_Prediction import K_Modulus
# from models.elastic.K_PredictionV2 import K_ModulusV2
# from models.elastic.K_Pred_FullInput import K_Modulus_Full
# from models.elastic.K_Classficiation import K_Modulus_Full
from models.elastic.Kratz import Kratz
if __name__ == "__main__":

    device = "cpu" #torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")


    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    model = Kratz()
    # model.load_state_dict(torch.load('model_weights/HuberCleanedMassive.pth', weights_only=True))
    # model = K_Modulus_Full()
    # model.load_state_dict(torch.load('model_weights/HuberLoss.pth', weights_only=True))


    # Run the optimzer
    model = run_optimizer(config, Kratz, model=model)

    # model = K_ModulusV2()
    # model.load_state_dict(torch.load('model_weights/KPredictionV2.pth', weights_only=True))


    # for layer in model.children():
    #     if isinstance(layer, nn.Linear):
    #         print(layer.state_dict()['weight'])
    #         print(layer.state_dict()['bias'])
    # print(str(summary(model, input_size=(100, 40, 2))))
    evaluate_directory(model, config, input_type="coordinates")
    # prediction = evaluate_single(model, "data/test_images/2083.png")
    # print(prediction)

    # prediction_tensor = evaluate_directory(model, )
