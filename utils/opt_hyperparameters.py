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
import matplotlib.pyplot as plt
# import torchvision.models as models
# from torchinfo import summary

from utils.optimizer import run_optimizer
from utils.evaluation import evaluate_directory
from utils.extraction import PendantDropDataset, extract_data_paths
import utils.check_config as check_config

# from models.simple.image_input.five_layer import FiveLayerCNN
# from models.simple.image_input.grayscaletransform import GrayscaleTransform
# from models.simple.rz_input.Xanathor import Xanathor
# from models.elastic.elasticbasic import Elastic
# from models.elastic.Gandalf import Gandalf
from models.elastic.Empty import Empty


def create_cross_validation(master_dataset, k_folds):
    """
    Generator that yields a k divided dataset.
    """
    k_datasets = master_dataset.split_k_dataset(k_folds)


    for i in range(k_folds):
        training_indexes = list(range(k_folds))
        training_indexes.remove(i)
        training_set = master_dataset.combine_datasets([k_datasets[i] for i in training_indexes])
        testing_set = k_datasets[i]
        yield training_set, testing_set



def optimize_hyperparams(modelType, config):


    CHOSEN_MODEL = modelType

    check_config.check_all_data_paths()
    check_config.check_hyperparams()

    data_paths = config["data_paths"]
    settings = config["settings"]
    hyper_params = config["hyper_param"]
    k_folds= hyper_params["k_folds"]
    if config["training_parameters"]["visualize_training"]:
        x = input("Recommended to set visualize_training to false when optimizing hyperparameters. Continue regardless? (y/N): ")
        if x.lower() != "y":
            exit()

    params, rz, images, sigmas = extract_data_paths(data_paths)
    master = PendantDropDataset(params, rz, images, 
                            sigma_dir=sigmas, ignore_images=settings["ignore_images"], clean_data=True)

    LEARNING_RATES = hyper_params["learning_rates"]

    train_losses = []
    test_losses = []
    
    for lr in LEARNING_RATES:
        print(f"##### Beginning Learning Rate:: {lr} #####")
        cross_set = create_cross_validation(master, k_folds)
        lr_train = 0
        lr_test = 0
        for fold in range(k_folds):
            model = CHOSEN_MODEL()
            training_set, testing_set = next(cross_set)
            # Run the optimzer
            model, (ftrain_loss, ftest_loss) = run_optimizer(config, model=model, chosen_training=training_set, chosen_testing=testing_set, return_loss=True, chosen_learning=lr)
            lr_train += ftrain_loss
            lr_test += ftest_loss
            print(f"Fold #{fold}::= Training Loss: {ftrain_loss}, Testing Loss: {ftest_loss}")
        train_losses.append(lr_train / k_folds)
        test_losses.append(lr_test / k_folds)
        print(f"Completed learning rate {lr}, with average training loss {train_losses[-1]}, average testing loss {test_losses[-1]}")

    

    plt.plot(LEARNING_RATES, train_losses, c="red", label="Train Losses")
    plt.xlabel("Learning Rates")
    plt.plot(LEARNING_RATES, test_losses, c="blue", label="Test Losses")
    plt.ylabel("Testing and Training Loss")
    plt.legend()
    plt.title("Learning Rate Optimization")
    plt.savefig("LearningRateOptGandLinear" + ".png")
    plt.show()


    plt.plot(LEARNING_RATES, train_losses, c="red", label="Train Losses")
    plt.xlabel("Learning Rates (Log)")
    plt.xscale("log")
    plt.plot(LEARNING_RATES, test_losses, c="blue", label="Test Losses")
    plt.ylabel("Testing and Training Loss")
    plt.legend()
    plt.title("Learning Rate Optimization")
    plt.savefig("LearningRateOptGandLog" + ".png")
    plt.show()

