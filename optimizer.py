# Runs model training and testing
# Modified from PyTorch Optimization tutorial

import torch
from utils.dataloader import PendantDataLoader
from model import PendantNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils.extraction import PendantDropDataset
import torchvision.models as models
import math

from five_layer import FiveLayerCNN
from single_layer import SingleLayerCNN



def train_loop(dataloader, model, loss_fxn, optimizer, batch_size):
    ## Uncomment for MNIST Fashion dataset
    # size = len(dataloader.dataset) 
    ## Uncomment for custom dataset
    size = len(dataloader.data)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(X.dtype)
        # print("Input Shape", X.shape)
        # print(y.dtype)
        # print("Expected output shape", y.shape)
        pred = model(X)
        # print("Prediction Shape", pred.shape)
        loss = loss_fxn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fxn, testing_size, num_batches, tolerance):
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fxn(pred, y).item()
            print("prediction shape", pred.shape)
            print("prediction value", pred)
            print("y shape", y.shape)
            print("y value", y)
            correct += (torch.isclose(pred, y, rtol=0, atol=tolerance)).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= testing_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




def run_optimizer(config_object, CNNModel):

    # Extracting information from config file (currently config object)
    data_paths = config_object["data_paths"]

    training_params = config_object["training_parameters"]

    learning_rate = training_params["learning_rate"]
    num_batches = training_params["num_batches"]
    epochs = training_params["epochs"]
    testing_size = training_params["testing_size"]
    random_seed = training_params["random_seed"]

    testing_params = config_object["testing_parameters"]

    test_num_batches = testing_params["num_batches"]

    ##############################################################
    ### Custom Modules
    ##############################################################

    drop_dataset = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"])
    training_data, testing_data = drop_dataset.split_dataset(testing_size, random_seed)

    batch_size = int(len(training_data)/ num_batches)

    train_dataloader = PendantDataLoader(training_data, num_batches=num_batches)
    test_dataloader = PendantDataLoader(testing_data, test_num_batches)


    model = CNNModel()

    loss_fxn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Training Model\n===============================")
    print([n for n, _ in model.named_children()])

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fxn, optimizer, batch_size)
        test_loop(test_dataloader, model, loss_fxn, testing_size, test_num_batches, config["testing_parameters"]["absolute_tolerance"])
    print("Done!")

    ### Save Model
    if config["save_info"]["save_model"]:
        torch.save(model.state_dict(), config_object["save_info"]["model_name"])
        print(f"Model weights saved to {config_object["save_info"]["save_model"]}")
    else:
        print(f"Model weights not saved")


