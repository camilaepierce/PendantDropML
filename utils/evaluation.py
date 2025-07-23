"""
Evaluation of model predictions.

Last modified: 6.26.2025
"""
from torch import Tensor, from_numpy, rand, isnan
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from utils.extraction import PendantDropDataset
from utils.visualize import scattershort


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
    isElastic = config_object["settings"]["isElastic"]
    isKMod = config_object["settings"]["calculateKMod"]

    if isKMod:
        from models.elastic.Extreme2 import Extreme
        from torch import load
        run_model = Extreme()
        run_model.load_state_dict(load('model_weights/HuberCleanedMassive.pth', weights_only=True))
    else:
        run_model = None
    

    drop_dataset = PendantDropDataset(data_paths["params"], data_paths["rz"], data_paths["images"], data_paths["sigmas"], ignore_images=(input_type!="image"), clean_data=True)

    evaluation_info = []
    nan_samples = []

    #for each sample
    for i, sample_id in enumerate(drop_dataset.available_samples):
        #save sample features {'image', 'coordinates', 'surface_tension', 'Wo_Ar'}.
        features = drop_dataset[sample_id]
        Wo = features["Wo_Ar"]["Wo"]
        Ar = features["Wo_Ar"]["Ar"]
        K = features["Wo_Ar"]["Kmod"]
        G = features["Wo_Ar"]["Gmod"]
        frac = features["Wo_Ar"]["frac"]

        if isElastic:
            sample_sigma = features["sigma_tensor"] # sample_sigma = true stress tensor
            act_sigma = features["surface_tension"] # act_sigma = initial surface tension
        else:
            sample_sigma = features["surface_tension"]

        #predict sample
        if input_type == "image":
            model_input = Tensor.float(from_numpy(np.array(features['image']))).unsqueeze(0)
        elif input_type == "coordinates":
            # print(features["coordinates"].shape)
            model_input = Tensor.float(from_numpy(features["coordinates"])).unsqueeze(0)
            # model_input = rand((4, 1, 2, 40))

        prediction = model(model_input).detach().numpy()     
        if np.any(np.isnan(prediction)):
            nan_samples.append(np.array([Wo, Ar, sample_sigma]))
        else:
            #calculate differences
            true_diff = sample_sigma - prediction # true stress tensor - predicted stress tensor
            true_diff_sq = np.square(true_diff)
            mse = np.average(true_diff_sq)
            relative_error = np.divide(np.absolute(true_diff), sample_sigma)
            #save info:: Wo, Ar, act_sigma, avg_sigma, avg_pred, true, relative, mse, K, G, frac
            if isElastic:
                evaluation_info.append(np.array([Wo, Ar, np.average(sample_sigma), np.average(prediction), np.average(true_diff), np.average(relative_error), mse, K, G, frac, act_sigma]))
            else:
                #save info:: Wo, Ar, sigma, pred, true, relative, mse
                evaluation_info.append(np.array([Wo, Ar, sample_sigma, prediction, true_diff, relative_error, mse]))

    #save data info to file
    evaluation_info = np.asarray(evaluation_info)
    print("nan samples:", nan_samples)

    if isElastic:
        np.savetxt(config_object["save_info"]["eval_results"] + "Evaluation.txt", evaluation_info, delimiter=",",
               header="Wo,Ar,sample_sigma,prediction,abs_error,rel_error,mse,K,G,frac,act_sigma")
    else:
        np.savetxt(config_object["save_info"]["eval_results"] + "Evaluation.txt", evaluation_info, delimiter=",",
               header="Wo,Ar,sample_sigma,prediction,abs_error,rel_error,mse")
    #save prediction info to file

    if visualize:

        plt.clf()


        # [Wo, Ar, sample_sigma, prediction, true_diff, relative_error]
        #  0    1       2           3           4           5           of evaluation_info
        all_Wo = evaluation_info[:, 0] # Worthington Number
        all_Ar = evaluation_info[:, 1] # Aspect Ratio
        all_sigma = evaluation_info[:, 2] # Elastic - True Stress Tensor
        all_pred = evaluation_info[:, 3] # Elastic - Predicted Stress Tensor
        all_true = evaluation_info[:, 4] # True Difference
        all_rel = evaluation_info[:, 5] # Relative Error
        all_mse = evaluation_info[:, 6] # Mean Squared Error
        if isElastic:
            all_K = evaluation_info[:, 7] # Dilatation Modulus (K)
            all_G = evaluation_info[:, 8] # Shear Modulus (G)
            all_frac = evaluation_info[:, 9] # Compression Fraction
            all_act = evaluation_info[:, 10] # Elastic - Initial Surface Tension

        with open(config_object["save_info"]["eval_results"] + "Distribution.txt", "a") as f:
            f.write("Sample Distribution\n")
            f.write(f"Worthington Number:: Mean: {np.mean(all_Wo)} Std Dev: {np.std(all_Wo)}\n")
            f.write(f"Aspect Ratio:: Mean: {np.mean(all_Ar)} Std Dev: {np.std(all_Ar)}\n")
            if isElastic:
                f.write(f"Stress Tensor:: Mean: {np.mean(all_sigma)} Std Dev: {np.std(all_sigma)}\n")
                f.write("Prediction Distribution\n")
                f.write(f"Stress Tensor:: Mean: {np.mean(all_pred)} Std Dev: {np.std(all_pred)}\n")


            else:
                f.write(f"Surface Tension:: Mean: {np.mean(all_sigma)} Std Dev: {np.std(all_sigma)}\n")
                f.write("Prediction Distribution\n")
                f.write(f"Surface Tension:: Mean: {np.mean(all_pred)} Std Dev: {np.std(all_pred)}\n")

            
        #plot data info, Wo vs Ar vs Surface Tension distribution
        # norm_all_sigma = Normalize()(all_sigma) #(all_sigma - np.min(all_sigma)) / (np.max(all_sigma) - np.min(all_sigma))

        plt.scatter(all_act, all_rel, c="slateblue", marker=".")
        plt.xlabel("Initial Surface Tension")
        plt.ylabel("Relative Error")
        plt.ylim(0, 1)
        plt.title("Initial Surface Tension vs Relative Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyRelIdealZoom" + ".png")

        plt.show(block=False)
        plt.clf()

        # def scattershort(xdata, ydata, cdata, cmap, 
        #                  xfull, xshort, 
        #                  yfull, yshort, 
        #                  cfull, cshort, 
        #                  config_object, norm=Normalize()):
        scattershort(all_Wo, all_Ar, all_act, "viridis", 
                    "Worthington Number", "Wo",
                    "Aspect Ratio", "Ar",
                    "Initial Surface Tension", "IST",
                    config_object, norm="log")
        
        scattershort(all_act, all_sigma, all_mse, "viridis", 
                    "Initial Surface Tension", "IST",
                    "True Stress Tensor (Average)", "TST",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        #plot Wo vs Ar vs accuracy
        # norm_all_true = Normalize()(all_true)
        scattershort(all_Wo, all_Ar, all_true, "plasma", 
                    "Worthington Number", "Wo",
                    "Aspect Ratio", "Ar",
                    "Absolute Error", "Abs",
                    config_object)
        
        # norm_all_rel = Normalize()(all_rel)
        scattershort(all_Wo, all_Ar, all_rel, "plasma", 
                    "Worthington Number", "Wo",
                    "Aspect Ratio", "Ar",
                    "Relative Error", "Rel",
                    config_object)
        
        scattershort(all_Wo, all_Ar, all_mse, "plasma", 
                    "Worthington Number", "Wo",
                    "Aspect Ratio", "Ar",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        scattershort(all_K, all_G, all_mse, "magma", 
                    "Dilitational Modulus (K)", "K",
                    "Shear Modulus (G)", "G",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        scattershort(all_K, all_frac, all_mse, "magma", 
                    "Dilitational Modulus (K)", "K",
                    "Compression Fraction", "Frac",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")

        scattershort(all_K, all_act, all_mse, "magma", 
                    "Dilatational Modulus (K)", "K",
                    "Initial Surface Tension", "IST",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        scattershort(all_sigma, all_K, all_mse, "magma", 
                    "True Stress Tensor", "TST",
                    "Dilatational Modulus (K)", "K",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        scattershort(all_sigma, all_Ar, all_mse, "magma", 
                    "True Stress Tensor", "TST",
                    "Aspect Ratio", "Ar",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        scattershort(all_sigma, all_Wo, all_mse, "magma", 
                    "True Stress Tensor", "TST",
                    "Worthington Number", "Wo",
                    "Mean Squared Error", "MSE",
                    config_object, norm="log")
        
        #plot Surface tension vs accuracy
        plt.scatter(all_act, all_true, c='forestgreen', marker=".")
        plt.xlabel("Initial Surface Tension")
        plt.ylabel("Absolute Error")
        plt.title("Surface Tension vs Absolute Error")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyTrue" + ".png")

        plt.show(block=False)
        plt.clf()


        plt.scatter(all_act, all_mse, c='forestgreen', marker=".")
        plt.xlabel("Initial Surface Tension")
        plt.ylabel("Mean Squared Error")
        plt.title("Surface Tension vs MSE")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyMSE" + ".png")

        plt.show(block=False)
        plt.clf()


        ### Plotting with the inclusion of average surface tension ###

        plt.scatter(all_act, all_true, c=all_sigma, norm=Normalize(), cmap='viridis', marker=".")
        plt.xlabel("Initial Surface Tension")
        plt.ylabel("Absolute Error")
        plt.title("Surface Tension vs Absolute Error")
        plt.colorbar(label=f"Stress Tensor Average")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyTrueAvgST" + ".png")

        plt.show(block=False)
        plt.clf()

        plt.scatter(all_act, all_rel, c=all_sigma, norm=Normalize(), cmap='viridis', marker=".")
        plt.xlabel("Surface Tension")
        plt.ylabel("Relative Error")
        plt.title("Surface Tension vs Relative Error")
        plt.colorbar(label=f"Stress Tensor Average")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyRelAvgST" + ".png")

        plt.show(block=False)
        plt.clf()

        plt.scatter(all_act, all_mse, c=all_sigma, norm=Normalize(), cmap='viridis', marker=".")
        plt.xlabel("Surface Tension")
        plt.xscale("log")
        plt.ylabel("Mean Squared Error")
        plt.title("Surface Tension vs MSE")
        plt.colorbar(label=f"Surface Tension Average")
        plt.savefig(config_object["save_info"]["eval_results"] + "SurfAccuracyMSEAvgST" + ".png")

        plt.show(block=False)
        plt.clf()

