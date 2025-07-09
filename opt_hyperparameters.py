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
from utils.extraction import PendantDropDataset

# from models.simple.image_input.five_layer import FiveLayerCNN
# from models.simple.image_input.grayscaletransform import GrayscaleTransform
# from models.simple.rz_input.Xanathor import Xanathor
# from models.elastic.elasticbasic import Elastic
from models.elastic.Gandalf import Gandalf
# from models.elastic.Empty import Empty


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



if __name__ == "__main__":

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    k_folds=10
    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)

    data_paths = config["data_paths"]
    settings = config["settings"]

    master = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], 
                                       sigma_dir=data_paths["sigmas"], ignore_images=settings["ignoreImages"])
    learning_rates = [.000001, .00001, .0001, .001, .005, .01, .05, .1, .2, .3, .5]

    train_losses = []
    test_losses = []
    
    for lr in learning_rates:
        print(f"##### Beginning Learning Rate:: {lr} #####")
        cross_set = create_cross_validation(master, k_folds)
        lr_train = 0
        lr_test = 0
        for fold in range(k_folds):
            model = Gandalf()
            training_set, testing_set = next(cross_set)
            # Run the optimzer
            model, (ftrain_loss, ftest_loss) = run_optimizer(config, Gandalf, model=model, chosen_training=training_set, chosen_testing=testing_set, return_loss=True)
            lr_train += ftrain_loss
            lr_test += ftest_loss
            print(f"Fold #{fold}::= Training Loss: {ftrain_loss}, Testing Loss: {ftest_loss}")
        train_losses.append(lr_train / k_folds)
        test_losses.append(lr_test / k_folds)
        print(f"Completed learning rate {lr}, with average training loss {train_losses[-1]}, average testing loss {test_losses[-1]}")

    

    plt.plot(learning_rates, train_losses, c="red")
    plt.xlabel("Learning Rates")
    plt.plot(learning_rates, test_losses, c="blue", )
    plt.ylabel("Testing and Training Loss")
    plt.show()

