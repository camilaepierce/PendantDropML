"""
Custom Dataset class and utility functions for dataset processing from local files.

Last modified: 6.26.2025
"""
import os
import json
import re
from skimage import io
import numpy as np
from torch.utils.data import  Dataset
import random
import warnings


warnings.filterwarnings("ignore")


def extract_data_frame_single(file):
    """
    Extracts rz data from test_data_rz files

    Parameters:
        file (str) : Path to file. Absolute or relative

    Returns:
        numpy tensor of drop frame
    """
    # Option for unpack=True, allows r, z = np.loadtxt(...)
    return np.loadtxt(file, delimiter=",")

def extract_data_frame_directory(dir_path):
    """
    Applies data extraction to every file in directory.

    Parameters:
        dir_path (str) : Path to directory. Absolute or relative.

    Returns:
        Dictionary of {sample_id (str): surface tension (float)}
    """
    return create_dictionary_per_file(dir_path, extract_data_frame_single)


def extract_surface_tension_single(file):
    """
    Extracts surface tension from parameter file

    Parameters:
        file (str) : Path to file. Absolute or relative

    Returns:
        Float of sample's sigma value.
    """
    return json.load(file)["sigma"]

def extract_surface_tension_directory(dir_path):
    """
    Extracts surface tension from every folder in given path.
    
    Paramters:
        dir_path (folder path)
    
    Returns:
        Dictionary of {sample_id (str): surface tension (float)}
    """
    def for_filelike_object(filename):
        return extract_surface_tension_single(open(filename))
    return create_dictionary_per_file(dir_path, for_filelike_object)

def extract_Wo_Ar_single(file):
    """Extracts Wo and Ar information from file object"""
    params = json.load(file)
    return {"Wo": params["Wo_paper"], "Ar" : params["Ar_paper"]}

def extract_Wo_Ar_directory(dir_path):
    """Extracts Wo and Ar information from dictionary path"""
    def for_filelike_object(filename):
        return extract_Wo_Ar_single(open(filename))
    return create_dictionary_per_file(dir_path, for_filelike_object)

def create_dictionary_per_file(dir_path, each_fxn):
    """
    Extracts information from each file in a directory.
    
    Parameters:
        dir_path (str) : Path (absolute or relative) to requested directory.
        each_fxn (function(str -> T)) : Function to apply to each file in order to extract desired information.

    Returns:
        Dictionary of {sample_id (str): each_fxn result (T)}
    """
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





class PendantDropDataset(Dataset):
    """Pendant Drop Dataset, inherits from torch.utils.data.Dataset Class."""

    def __init__(self, params_dir, rz_dir, img_dir, transform=None, select_samples=None, ignore_images=False):
        """
        Some good documentation.

        Parameters:
            params_dir (str) : Path (absolute or relative) towards directory with parameter (json) files
            rz_dir (str) : Path (absolute or relative) towards directory with rz coordinate (txt) files
            img_dir (str) : Path (absolute or relative) towards directory with image (png) files
            transform (bool) : Preprocessing data. NOTE Not implemented


        Returns:
            New Dataset Object

        TODO Need to construct a custom sampler (Dataset relies on integer indexing)
        """
        self.params_dir = params_dir
        self.rz_dir = rz_dir
        self.img_dir = img_dir
        self.coord_outline_dict = extract_data_frame_directory(rz_dir)
        self.surf_tens_dict = extract_surface_tension_directory(params_dir)
        self.Wo_Ar_dict = extract_Wo_Ar_directory(params_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.ignore_images = ignore_images
        # self.img_sigfigs = img_sigfigs
        if select_samples is None:
            self.available_samples = set(self.coord_outline_dict.keys())
        else:
            self.available_samples = select_samples

    def __len__(self):
        """ Returns size of dataset """
        return len(self.surf_tens_dict)
    
    def __getitem__(self, idx):
        """
        Gets specified sample from dataset.

        Parameters:
            idx (str) : Index / name of sample, requires idx to in the given directory

        Returns:
            Dictionary of requested sample, or Warning if idx not in the dataset. Dictionary has the keys {'image', 'coordinates', and 'surface_tension'}.
        """
        if (idx not in self.surf_tens_dict):
            return Warning(f"The requested sample_id {idx} does not exist. Did you match the name of a sample exactly?")
        img_name = os.path.join(self.img_dir, f"{idx}.png")

        if self.ignore_images:
            image = None
        else:
            image = io.imread(img_name)

        coords = self.coord_outline_dict[idx]

        sample = {'image': image, 'coordinates': coords, 'surface_tension': self.surf_tens_dict[idx], 'Wo_Ar' : self.Wo_Ar_dict[idx]} #
        ## Could choose whether or not to include rz coordinates vs image (?)

        return sample
    
    def split_dataset(self, k, random_seed=None):
        """
        Splits the dataset into two datasets of size {k} and size {len(this) - k}

        Parameters:
            k (int) : size of future testing set

        Returns:
            (TrainingDataset, TestingDataset) both of type PendantDropDataset
        """
        seeded_random = random.Random(random_seed)
        order = seeded_random.sample(list(self.available_samples), len(self.available_samples))
        testingset = PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, select_samples=order[:k], ignore_images=self.ignore_images)
        trainingset = PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, select_samples=order[k:], ignore_images=self.ignore_images)
        return (trainingset, testingset)

if __name__ == "__main__":

    ###################################################
    ### Test Case for evaluating single image
    ###################################################

    # n=2467

    # outline_frame = extract_data_frame_single(f"data/test_data_rz/rz{n:04d}.txt") #data/test_data_rz/rz2024.txt

    # img_name = f"data/test_images/{n:04d}.png"


    # print('Image name: {}'.format(img_name))
    # print('Coordinates shape: {}'.format(outline_frame.shape))
    # print('First 4 Coordinates: {}'.format(outline_frame[:4]))

    # plt.figure()
    # show_image_with_outline(io.imread(img_name), outline_frame)

    # plt.show()
    ##################################################
    ### Test Case for Dataset Class
    ##################################################

    drop_dataset = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images")


    for i, sample_id in enumerate(drop_dataset.available_samples):
        sample = drop_dataset[sample_id]

        print(i, sample['image'].shape, sample['surface_tension'], sample["coordinates"].shape)
