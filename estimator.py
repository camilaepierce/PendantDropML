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
from models.elastic.K_Pred_FullInput import K_Modulus_Full
import math

def toInput(npArray, run_model = None):
    if run_model is None:
        return Tensor.float(from_numpy(npArray)).unsqueeze(0)
    else:
        first_input = npArray
        model_output = run_model(Tensor.float(from_numpy(npArray)).unsqueeze(0)).detach().numpy()
        concat = np.concatenate((first_input, model_output))
        return Tensor.float(from_numpy(concat)).unsqueeze(0)

if __name__ == "__main__":


    # Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    data_paths = config["data_paths"]
    settings = config["settings"]
    isKMod = settings["calculateKMod"]

    
    if isKMod:
        # Load model (must be already created)
        model = K_Modulus_Full()
        model.load_state_dict(torch.load('model_weights/KPredictionV3.pth', weights_only=True))
        model.eval() # set so predictions do not affect training

        # Load model (must be already created)
        tens_model = Extreme()
        tens_model.load_state_dict(torch.load('model_weights/HuberCleanedMassive.pth', weights_only=True))
        tens_model.eval() # set so predictions do not affect training
    else:
        # Load model (must be already created)
        model = Extreme()
        model.load_state_dict(torch.load('model_weights/HuberCleanedMassive.pth', weights_only=True))
        model.eval() # set so predictions do not affect training

        tens_model = None

    # Load image (from name? from file? from directory?) -- from dataset object

    master = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], 
                                        sigma_dir=data_paths["sigmas"], ignore_images=settings["ignoreImages"], clean_data=True,)
    
    # has keys: {'image', 'coordinates', 'surface_tension', 'Wo_Ar', and 'sigma_tensor'}
    # drop = master["9"]
    extra_verbose = False
    verbose = False

    all_all_diff = []
    all_all_rel = []
    all_all_first = []
    all_all_second = []
    all_all_first_rel = []
    all_all_second_rel = []

    for drop in master:
        prediction = model(toInput(drop["coordinates"], tens_model)).detach().numpy()
        all_diff = []
        all_rel = []
        all_first = []
        all_second = []
        all_first_rel = []
        all_second_rel = []

        # if np.average(prediction) > 6 and np.average(prediction) < 10:
        #     verbose = True
        if isKMod:
            pred = prediction
            act = drop["Wo_Ar"]["Kmod"]
            diff = pred - act
            rel_err = abs(diff) / act
            if extra_verbose:
                print(f"Prediction: {pred} Actual: {act} Difference: {diff} Percent Error: {rel_err}")
            all_diff.append(diff)
            all_rel.append(rel_err)
        else:
            for pred, act in zip(prediction, drop["sigma_tensor"]):
                # if np.any(np.less(act, 0)):
                    # print("Negative tension", drop["sample_id"])
                diff = pred - act
                rel_err = abs(diff) / act
                if extra_verbose:
                    print(f"Prediction: {pred} Actual: {act} Difference: {diff} Percent Error: {rel_err}")
                all_diff.append(diff)
                all_rel.append(rel_err)

                all_first.append(diff[0])
                all_second.append(diff[1])
                all_first_rel.append(rel_err[0])
                all_second_rel.append(rel_err[1])
        # print(all_rel)
        this_mse = np.average(np.square(np.array(all_diff)))
        this_rel = np.average(np.array(all_rel))

        this_first = np.average(np.square(np.array(all_first)))
        this_second = np.average(np.square(np.array(all_second)))
        this_first_rel = np.average(np.array(all_first_rel))
        this_second_rel = np.average(np.array(all_second_rel))
        if verbose:
            print(f"{drop["sample_id"]} Mean Squared Error: {this_mse:.4}, Percent Rel Error: {this_rel:2.2%}, Actual: {act}, Pred: {pred}, Diff: [{this_first:2.3}, {this_second:2.3}], Rel: [{this_first_rel:2.2%}, {this_second_rel:2.2%}]")
        all_all_diff.append(this_mse)
        all_all_rel.append(this_rel)
        all_all_first.append(this_first)
        all_all_second.append(this_second)
        all_all_first_rel.append(this_first_rel)
        all_all_second_rel.append(this_second_rel)
        # verbose = False
    # print([(idx, x) for idx, x in enumerate(all_all_rel)])
    print(f"ALL SAMPLES MSE: {np.average(np.array(all_all_diff)):.4}, Percent Rel Error: {sum(all_all_rel) / len(all_all_rel):2.2%}")
    print(f"BY COLUMN: Diff [{np.average(np.array(all_all_first)):.4}, {np.average(np.array(all_all_second)):.4}] Rel [{np.average(np.array(all_all_first_rel)):.4%}, {np.average(np.array(all_all_second_rel)):.4%}]")





