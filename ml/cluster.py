from torch.utils.data import DataLoader, random_split
from .dataset_class import ODELIA_DATASET
import torch
from monai.networks.nets import DenseNet121

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    annotation_file = r""
    img_dir = r""
    save_checkpoint_path = r""

    dataset = ODELIA_DATASET(annotation_file=r"", img_dir=r"")

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=8)

    model = DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3, pretrained=False).to(device)

