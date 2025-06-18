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
from optimizer import run_optimizer
from five_layer import FiveLayerCNN


if __name__ == "main":


    # Run the optimzer


    config = {
        "save_info" : {"modelName" : "/model_weights/fiveLayerModelSecondRun.pth", "save_model" : True},
        "data_paths" : {"params" : "data/test_data_params", "rz" : "data/test_data_rz", "images" : "data/test_images"},
        "training_parameters" : {"learning_rate" : 1e-5, "num_batches" : 10, "epochs" : 2, "testing_size" : 20, "random_seed" : 4},
        "testing_parameters" : {"num_batches": 1, "absolute_tolerance" : 0.3}
    }

    run_optimizer(config, FiveLayerCNN)