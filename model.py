"""
Main file for running and training models.

Last modified: 6.26.2025
"""
from json import load
import torch

from utils.optimizer import run_optimizer
from utils.evaluation import evaluate_directory
from utils.run_at_exit import save_model
import utils.check_config as check_config
from utils.opt_hyperparameters import optimize_hyperparams
import atexit


###########################
###     Import Model    
###########################
# from models.simple.image_input.five_layer import FiveLayerCNN
# from models.simple.image_input.grayscaletransform import GrayscaleTransform
# from models.simple.rz_input.Xanathor import Xanathor
# from models.elastic.elasticbasic import Elastic
# from models.elastic.Gandalf import Gandalf
# from models.elastic.Empty import Empty
from models.elastic.MyPrecious import MyPrecious
# from models.elastic.K_Prediction import K_Modulus
# from models.elastic.K_PredictionV2 import K_ModulusV2
# from models.elastic.K_Pred_FullInput import K_Modulus_Full
# from models.elastic.K_Classficiation import K_Modulus_Full
# from models.elastic.Kratz import Kratz
# from models.elastic.MyNewModel import NewModel

if __name__ == "__main__":

    ###########################
    ###     DO NOT EDIT     
    ###########################
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    check_config.check_all_data_paths()

    ### Open config file as dictionary
    with open("config.json") as jsonFile:
        config = load(jsonFile)
    
    ###########################
    ###     Select Model    
    ###########################

    ### Choose model type             
    model = MyPrecious() #NOTE this line NEVER gets deleted, only changed

    ### Load model from save, modify only the filepath: 'model_weights/[Model Name].pth'
    ### Keeping this line commented out creates a new model, and weights will be saved to config["settings"]["model_name"]
    model.load_state_dict(torch.load('model_weights/MyPrecious.pth', weights_only=True))

    ### Sets exit behavior, do not change
    atexit.register(save_model, config, model)

    ### Find optimal hyperparameters (learning rate)
    ### Do NOT include parentheses after modelType (e.i. Extreme not Extreme())
    # optimize_hyperparams(MyPrecious, config)

    # print("begun training")
    ### Run the optimzer, only run if training the optimizer
    # model = run_optimizer(config, model=model)

    ### Again affect exit behavior, do not change
    atexit.unregister(save_model)

    ### Evaluate Model
    evaluate_directory(model, config, input_type="coordinates")


