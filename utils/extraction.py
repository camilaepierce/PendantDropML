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
    Extracts rz or surface tension tensor data from test_data_rz files

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
    return {"Wo": params["Wo_paper"], "Ar" : params["Ar_paper"], "Kmod" : params["Kmod"], "Gmod" : params["Gmod"], "frac" : params["frac"]}

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


def extract_data_paths(data_path_config):
    params = data_path_config["folder"] + data_path_config["params"]
    rz = data_path_config["folder"] + data_path_config["rz"]
    images = data_path_config["folder"] + data_path_config["images"]
    sigmas = data_path_config["folder"] + data_path_config["sigmas"]
    return params, rz, images, sigmas




class PendantDropDataset(Dataset):
    """Pendant Drop Dataset, inherits from torch.utils.data.Dataset Class."""

    def __init__(self, params_dir, rz_dir, img_dir, sigma_dir=None, transform=None, select_samples=None, ignore_images=False, clean_data=False):
        """
        Some good documentation.

        Parameters:
            params_dir (str) : Path (absolute or relative) towards directory with parameter (json) files
            rz_dir (str) : Path (absolute or relative) towards directory with rz coordinate (txt) files
            img_dir (str) : Path (absolute or relative) towards directory with image (png) files
            transform (bool) : Preprocessing data. NOTE Not implemented


        Returns:
            New Dataset Object
        """
        self.params_dir = params_dir
        self.rz_dir = rz_dir
        self.img_dir = img_dir
        self.sigma_dir = sigma_dir
        self.coord_outline_dict = extract_data_frame_directory(rz_dir)
        self.surf_tens_dict = extract_surface_tension_directory(params_dir)
        self.Wo_Ar_dict = extract_Wo_Ar_directory(params_dir)

        if sigma_dir:
            self.sigmas_dict = extract_data_frame_directory(sigma_dir)
            self.output_size = 80
        else:
            self.output_size = 1
            self.sigmas_dict = None
        self.img_dir = img_dir
        self.transform = transform
        self.ignore_images = ignore_images
        # self.img_sigfigs = img_sigfigs
        if select_samples is None:
            self.available_samples = set(self.coord_outline_dict.keys())
        else:
            self.available_samples = select_samples.copy()
        
        if clean_data:
            self.clean_data()
            


    def __len__(self):
        """ Returns size of dataset """
        return len(self.available_samples)
    
    def __getitem__(self, idx):
        """
        Gets specified sample from dataset.

        Parameters:
            idx (str) : Index / name of sample, requires idx to in the given directory

        Returns:
            Dictionary of requested sample, or Warning if idx not in the dataset. Dictionary has the keys {'image', 'coordinates', 'surface_tension', 'Wo_Ar', and 'sigma_tensor'}.
        """
        if (idx not in self.surf_tens_dict):
            return Warning(f"The requested sample_id {idx} does not exist. Did you match the name of a sample exactly?")
        img_name = os.path.join(self.img_dir, f"{idx}.png")

        if self.ignore_images:
            image = None
        else:
            image = io.imread(img_name)
        if self.sigmas_dict:
            sigma_tensor = self.sigmas_dict[idx]
        else:
            sigma_tensor = None

        coords = self.coord_outline_dict[idx]

        sample = {'image': image, 'coordinates': coords, 'surface_tension': self.surf_tens_dict[idx], 
                  'Wo_Ar' : self.Wo_Ar_dict[idx], "sigma_tensor": sigma_tensor, "sample_id": idx} #
        ## Could choose whether or not to include rz coordinates vs image (?)

        return sample
    
    def __iter__(self):
        for sample_id in self.available_samples:
            yield self.__getitem__(sample_id)
    
    def split_dataset(self, test_size, random_seed=None):
        """
        Splits the dataset into two datasets of size {test_size} and size {len(this) - k}

        Parameters:
            test_size (int) : size of future testing set

        Returns:
            (TrainingDataset, TestingDataset) both of type PendantDropDataset
        """
        seeded_random = random.Random(random_seed)
        order = seeded_random.sample(list(self.available_samples), len(self.available_samples))
        trainingset = PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, sigma_dir=self.sigma_dir, select_samples=order[test_size:], ignore_images=self.ignore_images)
        testingset = PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, sigma_dir=self.sigma_dir, select_samples=order[:test_size], ignore_images=self.ignore_images)
        return (trainingset, testingset)
    
    def split_k_dataset(self, k, random_seed=None):
        """
        Splits the dataset into k datasets of size {len(test) // k}

        Parameters:
            test_size (int) : size of future testing set

        Returns:
            (TrainingDataset, TestingDataset) both of type PendantDropDataset
        """
        seeded_random = random.Random(random_seed)
        order = seeded_random.sample(list(self.available_samples), len(self.available_samples))

        all_sets = []
        split_order = np.array_split(order, k)
        for i in range(k):
            next_set = PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, sigma_dir=self.sigma_dir, select_samples=set(split_order[i]), ignore_images=self.ignore_images)
            all_sets.append(next_set)
        return all_sets
    
    def combine_datasets(self, sets_array):
        """
        Datasets must refer to the same folders
        """
        all_samples_combined = set()
        for dataset in sets_array:
            all_samples_combined.update(dataset.available_samples.copy())

        return PendantDropDataset(self.params_dir, self.rz_dir, self.img_dir, sigma_dir=self.sigma_dir, select_samples=all_samples_combined, ignore_images=self.ignore_images)
    
    def clean_data(self):
        """
        Removes drops that have a surface tension below 1.
        """
        for sample_id, surf_tens in self.surf_tens_dict.items():
            if surf_tens < 1:
                self.available_samples.remove(sample_id)
            if np.average(self.sigmas_dict[sample_id]) < 0:
                self.available_samples.remove(sample_id)



# class SingleDropSimple:
#     """
    
#     """

#     def __init__(self, sample_num, params_dir, rz_dir, img_dir, transform=None, ignore_images=False):
#         pass

#     def get_rz(self):
#         pass

#     def get_image(self):
#         pass

#     def get_sigma(self):
#         pass

# class SingleDropElastic(SingleDropSimple):

#     def __init__(self, params_dir, rz_dir, img_dir, sigma_dir, transform=None, ignore_images=False):
#         super.__init__(self, params_dir, rz_dir, img_dir, transform=transform, ignore_images=ignore_images)
#         self.sigma_dir = sigma_dir


#     def get_sigma_tensor(self):

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

    dirty_data = PendantDropDataset("data/elastic_massive/test_data_params", "data/elastic_massive/test_data_rz","data/elastic_massive/test_images", "data/elastic_massive/test_data_sigmas", ignore_images=True, clean_data=False)

    clean_data = PendantDropDataset("data/elastic_massive/test_data_params", "data/elastic_massive/test_data_rz","data/elastic_massive/test_images", "data/elastic_massive/test_data_sigmas", ignore_images=True, clean_data=True)


    print(len(dirty_data))
    print(len(clean_data))

    print()
    set_diff = dirty_data.available_samples.difference(clean_data.available_samples)
    print(set_diff, "difference")

    # for sample in set_diff:
    #     print(dirty_data[sample]["surface_tension"])


    # for i, sample_id in enumerate(dirty_data.available_samples):
    #     tens = dirty_data[sample_id]["surface_tension"]
    #     # if tens < 1:
    #     print(sample_id, tens)
    #     if i > 1000:
            # break
    # for i, sample_id in enumerate(drop_dataset.available_samples):
    #     sample = drop_dataset[sample_id]

        # print(i, "sample ID", sample_id, "surface_tension", sample['surface_tension'], "coords size", sample["coordinates"].shape,"tensor_size", sample["sigma_tensor"].shape)
    
        # Removed sample['image'].shape from test statement

    # print(drop_dataset["38"]["coordinates"])

    # print(type(drop_dataset["38"]["sigma_tensor"]))
    # print(drop_dataset["38"]["sigma_tensor"].shape)
