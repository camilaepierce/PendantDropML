"""
Manages model training and testing.
Modified from PyTorch Optimization tutorial.

Last modified: 6.26.2025
"""
import torch
from torch import nn
from torchinfo import summary

from utils.dataloader import PendantDataLoader
from utils.extraction import PendantDropDataset
from utils.visualize import plot_loss_evolution



def train_loop(dataloader, model, loss_fxn, optimizer, batch_size, train_losses, filename):
    """ Training loop for optimization. """
    size = len(dataloader.data)
    out = ""
    train_loss_avg = 0
    clip = .6
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # with open(filename, "a", encoding="utf-8") as f:
        #     f.write(str(X))
        # print(X[-1])
        # exit()
        X = torch.nan_to_num(X)
        if torch.isnan(X).any():
            print("NaN found")
        pred = model(X)
        # print("Prediction Shape", pred.shape)
        loss = loss_fxn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            out += (f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_loss_avg += loss
        else:
            train_loss_avg += loss.item()
    # print(Tensor.float())
    train_losses.append(train_loss_avg / (batch + 1))
    with open(filename, "a", encoding="utf-8") as f:
        f.write(out + "\n")



def test_loop(dataloader, model, loss_fxn, testing_size, num_batches, tolerance, test_losses, filename):
    """ Testing loop of optimization. """
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fxn(pred, y).item()
            with open(filename, "a", encoding="utf-8") as f:
                # for (pred_val, y_val) in zip(pred, y):
                #     f.write(f"Acutal: {y_val:3.3f} Estimate: {pred_val:3.3f} Difference: {(pred_val - y_val):3.3f}\n")
                correct += (torch.isclose(pred, y, rtol=0, atol=tolerance)).type(torch.float).sum().item()
                f.write(f"Actual Mean: {torch.mean(y)} Actual Std Dev: {torch.std(y)}\n")
                f.write(f"Prediction Mean: {torch.mean(pred)} Prediction Std Dev: {torch.std(pred)}\n")
    test_loss /= num_batches
    correct /= testing_size
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_losses.append(test_loss)




def run_optimizer(config_object, CNNModel, model=None):
    """
    Runs optimization of NN Model. Saves output to text file, saves training and testing loss progression to image file.

    Parameters:
        config_object (dict) : config object, created from config.json file modification
        CNNModel (class) : chosen model for creation or use
        model (nn.Model subclass) : optional, pretrained model for continued training
    
    Returns:
        Model with trained or updated weights.
    """

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

    results_file = config_object["save_info"]["results"] + ".txt"

    train_losses = []
    test_losses = []

    ##############################################################
    ### Custom Modules
    ##############################################################

    drop_dataset = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], ignore_images=True)
    training_data, testing_data = drop_dataset.split_dataset(testing_size, random_seed)

    batch_size = int(len(training_data)/ num_batches)

    train_dataloader = PendantDataLoader(training_data, num_batches=num_batches, feat_fxn=lambda x: x["coordinates"])
    test_dataloader = PendantDataLoader(testing_data, test_num_batches, feat_fxn=lambda x: x["coordinates"])

    if model == None:
        model = CNNModel()


    loss_fxn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with open(results_file, "a", encoding="utf-8") as f:
        f.write("Training Model\n===============================\n")
        f.write(str(summary(model, input_size=train_dataloader.feature_shape)) + "\n")

    for t in range(epochs):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"Epoch {t+1}\n-------------------------------\n")
        train_loop(train_dataloader, model, loss_fxn, optimizer, batch_size, train_losses, results_file)
        test_loop(test_dataloader, model, loss_fxn, testing_size, 
                  test_num_batches, config_object["testing_parameters"]["absolute_tolerance"], test_losses, results_file)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("Done!\n")
    print("Done!")

    plot_loss_evolution(epochs, train_losses, test_losses, config_object["save_info"]["results"], "MSE", save=True)
    
    ### Save Model
    with open(results_file, "a", encoding="utf-8") as f:
        if config_object["save_info"]["save_model"]:
            torch.save(model.state_dict(), config_object["save_info"]["modelName"])
            f.write(f"Model weights saved to {config_object["save_info"]["modelName"]}\n")
        else:
            f.write("Model weights not saved\n")
    print(train_losses)
    print(test_losses)
    return model
