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
    model.load_state_dict(torch.load('model_weights/ExtremeFinal.pth', weights_only=True))
    model.eval() # set so predictions do not affect training

    # Load image (from name? from file? from directory?) -- from dataset object

    master = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], 
                                        sigma_dir=data_paths["sigmas"], ignore_images=settings["ignoreImages"])
    
    # has keys: {'image', 'coordinates', 'surface_tension', 'Wo_Ar', and 'sigma_tensor'}
    drop = master["85"]

    prediction = model(toInput(drop["coordinates"])).detach().numpy()
    all_diff = []

    for pred, act in zip(prediction, drop["sigma_tensor"]):
        diff = pred - act
        print(f"Prediction: {pred} Actual: {act} Difference: {diff}")
        all_diff.append(diff)
    print()
    print(f"Mean Squared Error: {np.average(np.square(np.array(all_diff)))}")




