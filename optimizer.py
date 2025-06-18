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



learning_rate = 1e-5
num_batches = 10
epochs = 2
testing_size = 20

##############################################################
### Custom Modules
##############################################################

drop_dataset = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images")
training_data, testing_data = drop_dataset.split_dataset(testing_size, 4)

batch_size = int(len(training_data)/ num_batches)

train_dataloader = PendantDataLoader(training_data, num_batches=num_batches)
test_dataloader = PendantDataLoader(testing_data, 1)


model = FiveLayerCNN()

###############################################################
### MNIST Fashion Dataset
###############################################################

# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

# train_dataloader = DataLoader(training_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork()
########################################################################

loss_fxn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fxn, optimizer):
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


def test_loop(dataloader, model, loss_fxn):
    model.eval()

    ## Uncomment for MNIST Fashion dataset
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    ## Uncomment for custom dataset
    size = testing_size
    num_batches = 1 #math.floor(size / batch_size)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fxn(pred, y).item()
            print("prediction shape", pred.shape)
            print("prediction value", pred)
            print("y shape", y.shape)
            print("y value", y)
            correct += (torch.isclose(pred, y, rtol=0, atol=0.3)).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print("Entering training")
    train_loop(train_dataloader, model, loss_fxn, optimizer)
    print("Entering testing")
    test_loop(test_dataloader, model, loss_fxn)
print("Done!")

####################################
### Save Model
####################################

torch.save(model.state_dict(), 'fiveLayerModelWeights.pth')