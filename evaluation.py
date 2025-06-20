import torch
from utils.dataloader import PendantDataLoader
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils.extraction import PendantDropDataset
from torch import Tensor, from_numpy
import torchvision.models as models
import math
from skimage import io

from models.five_layer import FiveLayerCNN
from models.single_layer import SingleLayerCNN


def evaluate_single(model, image_path):
    """
    Gets surface tension prediction of image.

    Parameters:
        model (nn.Module) : pre-trained model
        image_path (str) : pendant drop image to evaluate

    Returns:
        Prediction of surface tension.
    """
    # Set model mode to eval
    model.eval()
    image = Tensor.float(from_numpy(io.imread(image_path))).unsqueeze(0)

    prediction = model(image)
    return prediction


def evaluate_directory(model, eval_config):
    """
    Gets surface tension prediction of all images in a directory.

    Parameters:
        model (nn.Module) : pre-trained model
        eval_config (dict) : eval_config section of config file directory 

    Returns:
        Pytorch tensor of predictions.
    """
    # Set model mode to eval
    model.eval()
