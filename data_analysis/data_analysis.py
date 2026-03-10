import nibabel as nib
from pathlib import Path
import numpy as np
from PIL import Image

def number_of_channels(path_to_nii_gz):
    img = nib.load(path_to_nii_gz)
    return img.shape[2]

def unique_number_of_channels_in_folder(path_to_folder, dic):
    folder = Path(path_to_folder)
    for file in folder.iterdir():
        num_channel = number_of_channels(file)
        if num_channel in dic:
            dic[num_channel] += 1
        else: 
            dic[num_channel] = 1
    return dic

def unique_number_of_channels_in_dataset(path_to_dataset_folder):
    dataset_folder = Path(path_to_dataset_folder)
    dic = dict()
    for folder in dataset_folder.iterdir():
        unique_number_of_channels_in_folder(folder, dic)
    return dic

def folder_to_arr_lst(folder_path):
    folder = Path(folder_path)
    arr_lst = []
    for file in folder.iterdir():
        img = nib.load(file)
        img_data = img.get_fdata()
        arr_lst.append(img_data.T)
    return arr_lst

def stack_all_images_in_folder_and_save(parent_folder_path, save_folder_path):
    parent_folder = Path(parent_folder_path)
    for folder in parent_folder.iterdir():
        arr_lst = folder_to_arr_lst(folder)
        stacked = np.stack(arr_lst)
        np.save(f"{save_folder_path}\{folder.stem}.npy", stacked)

if __name__ == "__main__":
    pass