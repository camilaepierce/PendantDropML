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
Working on a more streamlined method of running models, but currently can be run in the command line through:

**Linux**
```
  > python optimizer.py
```

**Windows**
```
  > py optimizer.py
```
or by opening the folder in Visual Studio Code and running `optimzer.py`. Configurations can be specified at the top of `optimizer.py`, such as which model to run, learning rate, data folders, and number of batches.




**IMPORTANT** If you start running the optimizer and need to end it, you can just hit Ctrl+C (Keyboard Interrupt) to end the script midway

