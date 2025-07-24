"""
Manages model training and testing.
Modified from PyTorch Optimization tutorial.

Last modified: 6.26.2025
"""
import torch
from torch import nn
# from torchinfo import summary

from utils.dataloader import PendantDataLoader
from utils.extraction import PendantDropDataset, extract_data_paths
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
        if batch % 6 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            out += (f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
            train_loss_avg += loss
        else:
            train_loss_avg += loss.item()
    # print(Tensor.float())
    train_losses.append(train_loss_avg / dataloader.num_batches)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(out + "\n")



def test_loop(dataloader, model, loss_fxn, num_batches, tolerance, test_losses, filename):
    """ Testing loop of optimization. """
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fxn(pred, y).item()
            # with open(filename, "a", encoding="utf-8") as f:
            # for (pred_val, y_val) in zip(pred, y):
            #     f.write(f"Acutal: {y_val:3.3f} Estimate: {pred_val:3.3f} Difference: {(pred_val - y_val):3.3f}\n")
            correct += (torch.isclose(pred, y, rtol=0, atol=tolerance)).type(torch.float).sum().item()
                # f.write(f"Actual Mean: {torch.mean(y)} Actual Std Dev: {torch.std(y)}\n")
                # f.write(f"Prediction Mean: {torch.mean(pred)} Prediction Std Dev: {torch.std(pred)}\n")
    test_loss /= num_batches
    correct /= (len(dataloader.order) * dataloader.data.output_size) # total number of items isclose is counting
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_losses.append(test_loss)




def run_optimizer(config_object, CNNModel, model=None, chosen_training=None, chosen_testing=None, return_loss=False, chosen_learning=None):
    """
    Runs optimization of NN Model. Saves output to text file, saves training and testing loss progression to image file.

    Parameters:
        config_object (dict) : config object, created from config.json file modification
        CNNModel (class) : chosen model for creation or use
        model (nn.Model subclass) : optional, pretrained model for continued training
    
    Returns:
        Model with trained or updated weights.
    """
    ###########################################
    ###   Extracting Config Information     ###
    ###########################################
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

    ###########################################
    ###        Processing Settings          ###
    ###########################################
    settings = config_object["settings"]
    if settings["is_elastic"] and not settings["calculateKMod"]:
        labels_fxn = lambda x : x["sigma_tensor"]
    elif settings["calculateKMod"]:
        labels_fxn = lambda x : x["Wo_Ar"]["Kmod"]
    else:
        labels_fxn = lambda x : x["surface_tension"]

    if settings["ignoreImages"]:
        features_fxn = lambda x : x["coordinates"]
    else:
        features_fxn = lambda x : x["image"]

    ##############################################################
    ### Custom Modules
    ##############################################################
    if chosen_training == None or chosen_testing == None:
        params, rz, images, sigmas = extract_data_paths(data_paths)
        drop_dataset = PendantDropDataset(params, rz, images, 
                                        sigma_dir=sigmas, ignore_images=settings["ignoreImages"], clean_data=True)
        training_data, testing_data = drop_dataset.split_dataset(testing_size, random_seed)


        if (len(training_data.available_samples) == 0 or len(testing_data.available_samples) == 0):
            raise IndexError("You have only provided " + str(len(drop_dataset)) + " samples. Please update the config file.")
        
        
    else:
        training_data, testing_data = chosen_training, chosen_testing
    
    batch_size = int(len(training_data)/ num_batches)
    if (batch_size == 0):
            raise ValueError("You have only provided " + str(len(drop_dataset)) + " samples. Please update number of batches")
    if chosen_learning != None:
        learning_rate = chosen_learning

    
    train_dataloader = PendantDataLoader(training_data, num_batches=num_batches, feat_fxn=features_fxn, lab_fxn=labels_fxn)
    test_dataloader = PendantDataLoader(testing_data, test_num_batches, feat_fxn=features_fxn, lab_fxn=labels_fxn)

    if model == None:
        model = CNNModel()


    loss_fxn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with open(results_file, "a", encoding="utf-8") as f:
        f.write("Training Model\n===============================\n")
        # f.write(str(summary(model, input_size=train_dataloader.feature_shape, verbose=0)) + "\n")

    for t in range(epochs):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"Epoch {t+1}\n-------------------------------\n")
        train_loop(train_dataloader, model, loss_fxn, optimizer, batch_size, train_losses, results_file)
        test_loop(test_dataloader, model, loss_fxn, 
                  test_num_batches, config_object["testing_parameters"]["absolute_tolerance"], test_losses, results_file)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("Done!\n")
    # print("Done!")

    if training_params["visualize_training"]:
        plot_loss_evolution(epochs, train_losses, test_losses, config_object["save_info"]["results"], "MSE", save=True)
    

    ### Save Model
    with open(results_file, "a", encoding="utf-8") as f:
        if config_object["save_info"]["save_model"]:
            torch.save(model.state_dict(), config_object["save_info"]["modelName"])
            f.write(f"Model weights saved to {config_object["save_info"]["modelName"]}\n")
        else:
            f.write("Model weights not saved\n")
    if not return_loss:
        return model
    else:
        return model, (train_losses[-1], test_losses[-1])
