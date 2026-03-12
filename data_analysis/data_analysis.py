import nibabel as nib
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

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
        np.save(rf"{save_folder_path}\{folder.stem}.npy", stacked)

def concat_two_annotation_dataframes(df1, df2):
    df3 = pd.concat([df1, df2], ignore_index=True)
    return df3

if __name__ == "__main__":
    anno1 = r"C:\Users\magfa\Documents\ODELIA-2025\odelia_dataset\CAM\metadata_unilateral\annotation.csv"
    anno2 = r"C:\Users\magfa\Documents\ODELIA-2025\odelia_dataset\MHA\metadata_unilateral\annotation.csv"
    anno3 = r"C:\Users\magfa\Documents\ODELIA-2025\odelia_dataset\RUMC\metadata_unilateral\annotation.csv"
    anno4 = r"C:\Users\magfa\Documents\ODELIA-2025\odelia_dataset\UKA\metadata_unilateral\annotation.csv"

    df1, df2, df3, df4 = pd.read_csv(anno1), pd.read_csv(anno2), pd.read_csv(anno3), pd.read_csv(anno4)

    df5 = concat_two_annotation_dataframes(df1, df2)
    df6 = concat_two_annotation_dataframes(df5, df3)
    df7 = concat_two_annotation_dataframes(df6, df4)

    df7.to_csv(r"C:\Users\magfa\Documents\ODELIA-2025\annotation_CAM_MHA_RUMC_UKA.csv", index = False)