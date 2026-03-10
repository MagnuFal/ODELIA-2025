from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import decode_image

class ODELIA_DATASET(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.image_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transforms = transform

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[index, 0])
        image = decode_image(img_path)
        label = self.image_labels.iloc[index, -1]
        if self.transform:
            self.transform(image)
        return image, label