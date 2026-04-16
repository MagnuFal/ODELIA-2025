from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import decode_image
import numpy as np
import torch
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandZoomd

class ODELIA_DATASET(Dataset):
    def __init__(self, annotation_file, img_dir):
        self.image_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transforms = Compose([
            RandFlipd(keys = ["image"], prob=0.5),
            RandRotate90d(keys = ["image"], prob=0.5),
        ])

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0] + ".npy")
        arr = np.load(img_path)
        difference_in_scans = 8 - arr.shape[0]
        if difference_in_scans != 0:
            padding = np.repeat(arr[-1:, :, :, :], difference_in_scans, axis = 0)
            arr = np.concatenate([arr, padding], axis=0)

        if self.transforms:
            for i in range(arr.shape[0]):
                volume = arr[i, :, :, :]
                data = {"image" : volume}
                transform_data = self.transforms(data)
                arr[i, :, :, :] = np.asarray(transform_data["image"])

        image = torch.from_numpy(arr).float()
        label = self.image_labels.iloc[index, -1]
        label = torch.tensor(label, dtype=torch.long)
        return image, label, self.image_labels.iloc[index, 0]
    
class ODELIA_SKORCH_DATASET(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.image_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transforms = transform

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0] + ".npy")
        arr = np.load(img_path)
        difference_in_scans = 8 - arr.shape[0]
        if difference_in_scans != 0:
            padding = np.zeros((difference_in_scans, 32, 256, 256))
            arr = np.concatenate([arr, padding], axis=0)
        image = torch.from_numpy(arr).float()
        label = self.image_labels.iloc[index, -1]
        label = torch.tensor(label, dtype=torch.long)
        if self.transforms:
            image = self.transforms(image)
        return (image, label)