import os
import torch
import json
import pandas as pd
import re
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import warnings
warnings.filterwarnings("ignore")


def extract_data_frame_single(file):
    """
    Extracts rz data from test_data_rz files

    Parameters
        file (str) : path to file

    Returns
        numpy 
    """
    # Option for unpack=True, allows r, z = np.loadtxt(...)
    return np.loadtxt(file, delimiter=",")

def extract_data_frame_directory(dir_path):
    return create_dictionary_per_file(dir_path, extract_data_frame_single)


def extract_surface_tension_single(file):
    return json.load(file)["sigma"]

def extract_surface_tension_directory(dir_path):
    """Extracts surface tension from every folder in given path.
    
    Paramters:
        dir_path (folder path)
    
    Returns:
        Dictionary of {sample_id (str): surface tension (float)}
    """
    def for_filelike_object(filename):
        return extract_surface_tension_single(open(filename))
    return create_dictionary_per_file(dir_path, for_filelike_object)

def create_dictionary_per_file(dir_path, each_fxn):
    file_iter = os.scandir(dir_path)

    surf_dir = {}
    for file in file_iter:
        # Create a new dictionary entry for the sample, 
        sample_id = get_digits_from_filename(file.name)
        if sample_id in surf_dir:
            raise Exception("sample_id already in surf_dir, check dataset")
        surf_dir[sample_id] = each_fxn(file.path)

    return surf_dir

def get_digits_from_filename(filename):
    """ Extracts the digits from the filename of a file path or standalone name."""
    filename = filename.split('/')[-1]
    digits = re.findall(r'\d+', filename)

    return digits[0]

def show_image_with_outline(img, rz_frame):
    plt.imshow(img)
    plt.scatter(rz_frame[:, 0], rz_frame[:, 1], s=10, marker='.', c='r')
    # plt.pause(0.001)

class PendantDropDataset(Dataset):
    """Pendant Drop Dataset"""

    def __init__(self, params_dir, rz_dir, img_dir, img_sigfigs=4, transform=None):
        """
        Some good initialization documentation

        Parameters
        * params_dir (str)
        * rz_dir (str)
        * img_dir (str)
        * img_sigfigs (num)
        * transform (bool)


        Returns
            New Dataset Object
        """

        self.coord_outline_dict = extract_data_frame_directory(rz_dir)
        self.surf_tens_dict = extract_surface_tension_directory(params_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.img_sigfigs = img_sigfigs
        self.available_samples = set(self.coord_outline_dict.keys())

    def __len__(self):
        return np.size(self.coord_outline_dict)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist() ### Unecessary?

        # if (self.img_sigfigs == 3) :
        #     sample_id = f"{idx:03d}"
        # elif (self.img_sigfigs == 4):
        #     sample_id = f"{idx:04d}"
        # else:
        #     raise Exception("Please update code to reflect number of images")
        
        ## NEED TO CATCH if the sample does not exist
        if (idx not in self.surf_tens_dict):
            return Warning(f"The requested sample_id {idx} does not exist. Did you define the Dataset with the correct number of sigfigs?")
        img_name = os.path.join(self.img_dir, f"{idx}.png")
        print(img_name)
        print(idx)
        image = io.imread(img_name)

        coords = self.coord_outline_dict[idx]

        sample = {'image': image, 'coordinates': self.coord_outline_dict[idx], 'surface_tension': self.surf_tens_dict[idx]} #
        ## Could choose whether or not to include rz coordinates vs image (?)

        return sample


# n=2467

# outline_frame = extract_data_frame_single(f"data/test_data_rz/rz{n:04d}.txt") #data/test_data_rz/rz2024.txt

# img_name = f"data/test_images/{n:04d}.png"


# print('Image name: {}'.format(img_name))
# print('Coordinates shape: {}'.format(outline_frame.shape))
# print('First 4 Coordinates: {}'.format(outline_frame[:4]))

# plt.figure()
# show_image_with_outline(io.imread(img_name), outline_frame)

# plt.show()


# drop_dataset = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images", img_sigfigs=4)

# fig = plt.figure()


# for i, sample_id in enumerate(drop_dataset.available_samples):
#     sample = drop_dataset[sample_id]

#     print(i, sample['image'].shape, sample['surface_tension'], sample["coordinates"].shape)

#     ax = plt.subplot(1, 4, i+1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_image_with_outline(sample["image"], sample["coordinates"])

#     if i == 3:
#         plt.show()
#         break