import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision



def compare_to_params():
    pass



def show_image_with_outline(img, rz_frame):
    """Plots image on already created pyplot figure"""
    plt.imshow(img)
    plt.scatter(rz_frame[:, 0], rz_frame[:, 1], s=10, marker='.', c='r')
    # plt.pause(0.001)

def matplotlib_imshow(img, one_channel=False):
    """Modified from: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html"""
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# writer = SummaryWriter("model_weights")
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=True)

# writer.add_image('four_pendant_drop_images', img_grid)