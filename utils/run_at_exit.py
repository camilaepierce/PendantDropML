### Functions to be run at exit
import torch
from utils.evaluation import evaluate_directory

def save_model(config, model):
    print("Either an exception has occured, or you have chosen to terminate the program.")

    ### Handle saving model
    while True:
        x = input("Save model where it is? (y/N) ")
        if x.lower() in "yn":
            break
        print("Invalid response")

    if x.lower() == "y":
        print("Saving model")

        results_file = config["save_info"]["results_folder"] + config["settings"]["model_name"]
        model_save = config["save_info"]["model_weights_folder"] + config["settings"]["model_name"] + ".pth"

        with open(results_file, "a", encoding="utf-8") as f:
            torch.save(model.state_dict(), model_save)
            f.write(f"Model weights saved to {model_save}\n")
    else:
        print("Not saving model")

    ### Handle running evaluation with current model
    while True:
        x = input("Run evaluation? (y/N) ")
        if x.lower() in "yn":
            break
        print("Invalid response")

    if x.lower() == "y":
        print("Running evaluation")
        evaluate_directory(model, config, input_type="coordinates")
    else:
        print("Not evaluating model")
