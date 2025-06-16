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

from five_layer import FiveLayerCNN



learning_rate = 1e-3
batch_size = 4
epochs = 10
testing_size = 20

##############################################################
### Custom Modules
##############################################################

drop_dataset = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images")
training_data, testing_data = drop_dataset.split_dataset(testing_size, 4)

train_dataloader = PendantDataLoader(training_data, batch_size)
test_dataloader = PendantDataLoader(testing_data, batch_size)


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

loss_fxn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fxn, optimiser):
    ## Uncomment for MNIST Fashion dataset
    # size = len(dataloader.dataset) 
    ## Uncomment for custom dataset
    print(dataloader.data)
    size = len(dataloader.data)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
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
    size = len(dataloader.data)
    num_batches = dataloader.num_batches
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fxn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fxn, optimizer)
    test_loop(test_dataloader, model, loss_fxn)
print("Done!")