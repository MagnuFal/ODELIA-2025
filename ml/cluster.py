from torch.utils.data import DataLoader, random_split
from .dataset_class import ODELIA_DATASET
import torch
from monai.networks.nets import DenseNet121
from .optimization import optimizer_loop
from torchinfo import summary

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    annotation_file = r"/cluster/home/magnufal/TDT4265/annotation_CAM_MHA_RUMC_UKA.csv"
    img_dir = r"/cluster/home/magnufal/TDT4265/training_data"
    save_checkpoint_path = r"/cluster/home/magnufal/TDT4265/checkpoints/from_baseline_weights_with_momentum.pth"

    dataset = ODELIA_DATASET(annotation_file=annotation_file, img_dir=img_dir)

    val_percent = 0.1

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=8)

    model = DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3, pretrained=False).to(device)

    checkpoint = torch.load(r"/cluster/home/magnufal/TDT4265/checkpoints/baseline.pth", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    weights = torch.tensor([0.477046, 3.319444, 1.659722]).to(device)

    optimizer_loop(model=model, train_loader=train_loader, val_loader=val_loader, save_path=save_checkpoint_path, epochs=300, lr=1e-3,
                   momentum=0.79, nesterov=False)