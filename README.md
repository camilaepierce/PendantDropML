# PendantDropML
Clone this GitHub into an easily accessible folder.

## Setting Up the Virtual Environment
<details>
  <summary>Linux Setup</summary>

  ### Install `venv`
For best practices, set up a virtual environment to install packages locally. Requires installation of venv from python:
```
  $ apt install python3.12-venv
```
May require sudo (error message after attempting installation), if so:
```
  $ sudo apt install python3.12-venv
```
Then input the sudouser's login.

### Create Virtual Environment
Navigate to this repo's folder, then create the `.venv` directory.
```
  $ python -m venv .venv
```
Activate this virtual environment. In the same folder:
```
  $ source .venv/bin/activate
```
At this point, your terminal should look something like this:
```
  (.venv) (base) yourname@computer:~/path/to/your/folder/PendantDropML$
```

### Install Packages in `venv`
Use the `requirements.txt` file and pip to install required packages. From the command line still:
```
  $ pip install -r requirements.txt
```
All of the packages required to run this repo should now be downloaded to your virtual environment without affecting the rest of your computer!

### Deactivating the Virtual Environment
To deactivate the virtual environment, run in the command line:
```
  $ deactivate
```

</details>

<details>
  <summary>Windows Setup</summary>
  
  ### Install `venv`
For best practices, set up a virtual environment to install packages locally. Requires installation of venv from python:
```
  > winget install python3.12-venv
```
May require sudo (error message after attempting installation), if so:
```
  > sudo winget install python3.12-venv
```
Then input the sudouser's login.

### Create Virtual Environment
Navigate to this repo's folder, then create the `.venv` directory.
```
  > py -m venv .venv
```
Activate this virtual environment. In the same folder:
```
  > .venv/bin/activate
```
At this point, your terminal should look something like this:
```
  (.venv) C:\Users\yourname\path\to\your\folder>
```

### Install Packages in `venv`
Use the `requirements.txt` file and pip to install required packages. From the command line still:
```
  > pip install -r requirements.txt
```
All of the packages required to run this repo should now be downloaded to your virtual environment without affecting the rest of your computer!

</details>


## Running Models
The model.py file can be run directly in Visual Studio Code, or can be run via the command line. Any time you are using the command line, first please activate the virtual environment (installation above) by running `source .venv/bin/activate`.


**Linux**
```
  > python model.py
```

**Windows**
```
  > py model.py
```
or by opening the folder in Visual Studio Code and running `model.py`. Configurations can be specified at the top of `model.py`, such as which model to run, learning rate, data folders, and number of batches.

Customize the actions you want to take by commenting / uncommenting the desired lines. Current, train a model through running the optimizer. Load a model to continue training with the loading function. Evaluate a model through evaluate_directory(). Determine optimal hyperparameters (learning rate, etc.) through running the opt_hyperparameters script.

## Ending Abruptly
**IMPORTANT** If you start running the optimizer and need to end it, you can just hit Ctrl+C (Keyboard Interrupt) to end the script midway. If the script is ended during the optimizer script, you will see the following message:
```
[Some exception tracing message]
KeyboardInterrupt
Either an exception has occured, or you have chosen to terminate the program.
Save model where it is? (y/N)
```
and after the first response:
```
Saving model
Run evaluation? (y/N) y
Running evaluation
```
These will allow for a graceful exit of the program, without losing the progress in training.

## Recommended Config
A recommended config is saved within the RECOMMENDED.json file, and the following are recommended sizes for batches and training based on available sample data:


Sample      Training Batches    Testing Size    Testing Batches
~100            10                  7                   2         (mini)
~1,000          10                  100                 2
~10,000         30                  200                 4         (large)
~30,000         100                 600                10         (massive)

If you are training and you notice that your computer is significantly struggling, or (opening System Monitor), increase the number of batches so the program is taking smaller bites of the data.

## Creating a New Model
1. Make a copy of template.py in the models folder. Rename file and module name.
2. Modify sequential layer, recommended layers include Dropout or Normalization layers (including in the forward function), and in the sequential Linear layers and ReLU or LeakyReLU activations. See a full list of layers and their descriptions here: [text](https://docs.pytorch.org/docs/stable/nn.html)
3. Save model, make every layer's output matches the next layer's input.
4. At the top of `model.py`, add your import statement `from models.[your new file] import [your new model]`, and update the model type in the rest of the file.



<details>
<summary> Common Errors </summary>

1. IndexError: list index out of range
```
Traceback (most recent call last):
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/model.py", line 68, in <module>
    model = run_optimizer(config, model=model)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [...some more of the traceback...]
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/utils/extraction.py", line 98, in create_dictionary_per_file
    sample_id = get_digits_from_filename(file.name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/utils/extraction.py", line 111, in get_digits_from_filename
    return digits[0]
           ~~~~~~^^^
IndexError: list index out of range
```
There could be an extra file that is not part of a sample in your data folders.

2. Warning: sample_id X does not exist
```
                         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/utils/dataloader.py", line 43, in __iter__
    sample = self.data[sample_id]
             ~~~~~~~~~^^^^^^^^^^^
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/utils/extraction.py", line 185, in __getitem__
    raise Warning(f"The requested sample_id {idx} does not exist. Did you match the name of a sample exactly?")
Warning: The requested sample_id 1 does not exist. Did you match the name of a sample exactly?
```
There could be a half-saved sample. Check that the given sample_id exists in all relevant folders.

3. RuntimeError: Error(s) in loading state_dict for Model

```
Traceback (most recent call last):
  File "/home/camilapierce/Desktop/UNED/MLPendantDropUNED/model.py", line 57, in <module>
    model.load_state_dict(torch.load('model_weights/MyPrecious.pth', weights_only=True))
  File "/home/camilapierce/.pyenv/versions/3.12.11/lib/python3.12/site-packages/torch/nn/modules/module.py", line 2593, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for NewModel:
        Missing key(s) in state_dict: "linear_relu_stack.0.weight", "linear_relu_stack.0.bias", "linear_relu_stack.2.weight", "linear_relu_stack.2.bias", "linear_relu_stack.4.weight", "linear_relu_stack.4.bias". 
        Unexpected key(s) in state_dict: "linear_relu_stack.1.weight", "linear_relu_stack.1.bias", "linear_relu_stack.3.weight", "linear_relu_stack.3.bias". 
```
The shape of weights of the two models do not match. Check that you are loading the same model that you imported and are using.