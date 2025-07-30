from json import load
import os

with open("config.json") as jsonFile:
    config = load(jsonFile)

### Data Paths
def check_data_paths(data_path_config):
    assert(data_path_config["folder"][-1] == "/", "Please ensure the folder path is valid and ends with a '/'")
    assert(os.path.isdir(data_path_config["folder"]), "Folder does not exist")
    assert(os.path.isfile(data_path_config["folder"] + data_path_config["params"]), "File does not exist")
    assert(os.path.isfile(data_path_config["folder"] + data_path_config["rz"]), "File does not exist")
    if config["settings"]["is_elastic"]:
        assert(os.path.isfile(data_path_config["folder"] + data_path_config["sigmas"]), "File does not exist")
    if not config["settings"]["ignore_images"]:
        assert(os.path.isfile(data_path_config["folder"] + data_path_config["images"]), "File does not exist")

def check_all_data_paths():
    check_data_paths(config["data_paths"])
    check_data_paths(config["evaluation"]["data_paths"])


# Running Optimizer

def check_hyperparams():
    assert(not config["training_parameters"]["visualize_training"], "Please turn off visualize_training in your config file before running hyperparameter optimization.")
    assert(not config["save_info"]["save_model"], "Please turn off save_model in config before running hyperparameter optimization")