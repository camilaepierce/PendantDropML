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

def plot_loss_evolution(num_epochs, training_loss, testing_loss, model_name, loss_fxn, save=False):
    x_axes = range(1, num_epochs+1)
    plt.plot(x_axes, training_loss, c="darkorchid", label="Training Loss")
    plt.plot(x_axes, testing_loss, c="slateblue", label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({loss_fxn})")
    plt.xticks(x_axes)
    plt.title(model_name)
    plt.legend()

    if save:
        plt.savefig(model_name.replace(" ", "_") + ".png")
    
    plt.show()
# writer = SummaryWriter("model_weights")
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=True)

# writer.add_image('four_pendant_drop_images', img_grid)

if __name__ == "__main__":
    num_epochs = 20
    training_loss = [39, 36, 19, 20, 25, 19, 30, 28, 10, 17, 15, 14, 16, 16, 14, 17, 12, 9, 8, 10]
    testing_loss = [45, 40, 26, 22, 26, 16, 32, 29, 10, 19, 14, 16, 16, 17, 10, 12, 19, 11, 9, 10]
    print(len(training_loss))
    print(len(testing_loss))

    plot_loss_evolution(num_epochs, training_loss, testing_loss, "Grayscale Five Layer", "MSE")