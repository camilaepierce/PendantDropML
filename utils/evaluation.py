"""
Evaluation of model predictions.
NOTE catered to image as an input feature. Not yet able to process rz-coordinates.

Last modified: 6.26.2025
"""
from torch import Tensor, from_numpy, rand, isnan
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.extraction import PendantDropDataset
from models.five_layer import FiveLayerCNN
from models.single_layer import SingleLayerCNN


def evaluate_single(model, image_path):
    """
    Gets surface tension prediction of image.

    Parameters:
        model (nn.Module) : pre-trained model
        image_path (str) : pendant drop image to evaluate

    Returns:
        Prediction of surface tension.
    """
    # Set model mode to eval
    model.eval()
    image = Tensor.float(from_numpy(io.imread(image_path))).unsqueeze(0)

    prediction = model(image)
    return prediction


def evaluate_directory(model, config_object, visualize=True, input_type="image"):
    """
    Gets surface tension prediction of all images in a directory.

    Parameters:
        model (nn.Module) : pre-trained model
        eval_config (dict) : eval_config section of config file directory 

    Returns:
        Pytorch tensor of predictions.
    """
    # Set model mode to eval
    plt.clf()
    model.eval()
    data_paths = config_object["data_paths"]

    drop_dataset = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], ignore_images=(input_type!="image"))

    evaluation_info = []
    nan_samples = []

    #for each sample
    for i, sample_id in enumerate(drop_dataset.available_samples):
        #save sample features {'image', 'coordinates', 'surface_tension', 'Wo_Ar'}.
        features = drop_dataset[sample_id]
        Wo = features["Wo_Ar"]["Wo"]
        Ar = features["Wo_Ar"]["Ar"]
        sample_sigma = features["surface_tension"]
        #predict sample
        if input_type == "image":
            model_input = Tensor.float(from_numpy(np.array(features['image']))).unsqueeze(0)
        elif input_type == "coordinates":
            # print(features["coordinates"].shape)
            model_input = Tensor.float(from_numpy(features["coordinates"])).unsqueeze(0)
            # model_input = rand((4, 1, 2, 40))
        prediction = model(model_input).detach().numpy()     
        if np.isnan(prediction):
            nan_samples.append(np.array([Wo, Ar, sample_sigma]))
        else:
            #calculate differences
            true_diff = sample_sigma - prediction
            relative_error = np.absolute(true_diff) / sample_sigma
            #save info
            evaluation_info.append(np.array([Wo, Ar, sample_sigma, prediction, true_diff, relative_error]))


    #save data info to file
    evaluation_info = np.asarray(evaluation_info)
    print(nan_samples)

    np.savetxt(config_object["save_info"]["eval_results"] + "Evaluation.txt", evaluation_info, delimiter=",",
               header="Wo,Ar,sample_sigma,prediction,abs_error,rel_error")
    #save prediction info to file

    if visualize:

        plt.clf()


        # [Wo, Ar, sample_sigma, prediction, true_diff, relative_error]
        #  0    1       2           3           4           5           of evaluation_info
        all_Wo = evaluation_info[:, 0]
        all_Ar = evaluation_info[:, 1]
        all_sigma = evaluation_info[:, 2]
        all_pred = evaluation_info[:, 3]
        all_true = evaluation_info[:, 4]
        all_rel = evaluation_info[:, 5]

        with open(config_object["save_info"]["eval_results"] + "Distribution.txt", "a") as f:
            f.write("Sample Distribution\n")
            f.write(f"Worthington Number:: Mean: {np.mean(all_Wo)} Std Dev: {np.std(all_Wo)}\n")
            f.write(f"Aspect Ratio:: Mean: {np.mean(all_Ar)} Std Dev: {np.std(all_Ar)}\n")
            f.write(f"Surface Tension:: Mean: {np.mean(all_sigma)} Std Dev: {np.std(all_sigma)}\n")
            f.write("Prediction Distribution\n")
            f.write(f"Surface Tension:: Mean: {np.mean(all_pred)} Std Dev: {np.std(all_pred)}\n")

            
        #plot data info, Wo vs Ar vs Surface Tension distribution
        # norm_all_sigma = Normalize()(all_sigma) #(all_sigma - np.min(all_sigma)) / (np.max(all_sigma) - np.min(all_sigma))
        plt.scatter(all_Wo, all_Ar, c=all_sigma, norm="log", cmap="viridis", marker=".")
        plt.xlabel("Worthington Number (Wo)")
        plt.ylabel("Aspect Ratio (Ar)")
        plt.title("Training Data Wo vs Ar vs Surface Tension")
        plt.colorbar(label="Surface Tension")
        plt.savefig(config_object["save_info"]["eval_results"] + "WoArSurfTension" + ".png")

        plt.show(block=False)
        plt.clf()
        #plot Wo vs Ar vs accuracy
        # norm_all_true = Normalize()(all_true)
        plt.scatter(all_Wo, all_Ar, c=all_true, norm=Normalize(), cmap="plasma", marker=".")
        plt.xlabel("Worthington Number (Wo)")
        plt.ylabel("Aspect Ratio (Ar)")
        plt.title("Training Data Wo vs Ar vs AbsoluteError")
        plt.colorbar(label="Absolute Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "WoArAccuracyTrue" + ".png")

        plt.show(block=False)
        plt.clf()

        # norm_all_rel = Normalize()(all_rel)
        plt.scatter(all_Wo, all_Ar, c=all_rel, norm=Normalize(), cmap="plasma", marker=".")
        plt.xlabel("Worthington Number (Wo)")
        plt.ylabel("Aspect Ratio (Ar)")
        plt.title("Training Data Wo vs Ar vs RelativeError")
        plt.colorbar(label="Relative Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "WoArAccuracyRel" + ".png")

        plt.show(block=False)
        plt.clf()
 

        #plot Surface tension vs accuracy
        plt.scatter(all_sigma, all_true, c='forestgreen', marker=".")
        plt.xlabel("Surface Tension")
        plt.ylabel("Absolute Error")
        plt.title("Surface Tension vs Absolute Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyTrue" + ".png")

        plt.show(block=False)
        plt.clf()

        plt.scatter(all_sigma, all_rel, c="slateblue", marker=".")
        plt.xlabel("Surface Tension")
        plt.ylabel("Relative Error")
        plt.title("Surface Tension vs Relative Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyRel" + ".png")

        plt.show(block=False)
        plt.clf()



    
