from json import load
import torch
from torch import Tensor, from_numpy
import numpy as np

from utils.optimizer import run_optimizer
from utils.extraction import PendantDropDataset
from utils.dataloader import PendantDataLoader
# from models.simple.image_input.five_layer import FiveLayerCNN

# from models.elastic.elasticbasic import Elastic
# from models.elastic.Gandalf import Gandalf
# from models.elastic.Empty import Empty
from models.elastic.Extreme2 import Extreme

def toInput(npArray):
    return Tensor.float(from_numpy(npArray)).unsqueeze(0)

if __name__ == "__main__":


    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    data_paths = config["data_paths"]
    settings = config["settings"]

    
    # Load model (must be already created)
    model = Extreme()
    model.load_state_dict(torch.load('model_weights/HuberCleanedMassive.pth', weights_only=True))
    model.eval() # set so predictions do not affect training

    # Load image (from name? from file? from directory?) -- from dataset object

    master = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], 
                                        sigma_dir=data_paths["sigmas"], ignore_images=settings["ignoreImages"], clean_data=False)
    
    # has keys: {'image', 'coordinates', 'surface_tension', 'Wo_Ar', and 'sigma_tensor'}
    # drop = master["9"]
    extra_verbose = False
    verbose = False

    all_all_diff = []
    all_all_rel = []

    for drop in master:
        prediction = model(toInput(drop["coordinates"])).detach().numpy()
        all_diff = []
        all_rel = []
        if np.average(prediction) > 6 and np.average(prediction) < 10:
            verbose = True
        for pred, act in zip(prediction, drop["sigma_tensor"]):
            # if np.any(np.less(act, 0)):
                # print("Negative tension", drop["sample_id"])
            diff = pred - act
            rel_err = abs(diff) / act
            if extra_verbose:
                print(f"Prediction: {pred} Actual: {act} Difference: {diff} Percent Error: {rel_err}")
            all_diff.append(diff)
            all_rel.append(rel_err)
        this_mse = np.average(np.square(np.array(all_diff)))
        this_rel = np.average(np.array(all_rel))
        if verbose:
            print(f"Mean Squared Error: {this_mse:.4}, Percent Rel Error: {this_rel:2.2%}")
        all_all_diff.append(this_mse)
        all_all_rel.append(this_rel)
        verbose = False
    print(f"ALL SAMPLES MSE: {np.average(np.square(np.array(all_all_diff))):.4}, Percent Rel Error: {np.average(np.array(all_all_rel)):2.2%}")





