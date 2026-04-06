from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from skorch.callbacks import EpochScoring
from sklearn.model_selection import RandomizedSearchCV
from monai.networks.nets import DenseNet121
from torchinfo import summary
import torch
from .dataset_class import ODELIA_SKORCH_DATASET
from torch.utils.data import DataLoader

def create_net():
    return DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #annotation_file = r"/cluster/home/magnufal/TDT4265/annotation_CAM_MHA_RUMC_UKA.csv"
    #img_dir = r"/cluster/home/magnufal/TDT4265/training_data"
    annotation_file = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\annotation.csv"
    img_dir = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\RSH_np_arrays"
    save_checkpoint_path = r"C:\Users\magfa\Documents\ODELIA-2025\checkpoints\skorch_run_1.pth"
    #save_checkpoint_path = r"/cluster/home/magnufal/TDT4265/checkpoints/skorch_run_1.pth"

    dataset = ODELIA_SKORCH_DATASET(annotation_file=annotation_file, img_dir=img_dir)

    X_sl = SliceDataset(dataset, idx=0)
    y_sl = SliceDataset(dataset, idx=1)

    model = DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3).to(device)

    net = NeuralNetClassifier(module=create_net,
                              max_epochs = 120, # Same max_epochs, opt_mom and lr as baseline model
                              lr = 1e-3,
                              optimizer__momentum = 0,
                              verbose = 0,
                              train_split=False,)

    params = {
        "lr" : [0.001, 0.01, 0.05, 0.1],
        "optimizer__momentum" : [0, 0.5, 0.9],
        #"module__batchsize" : [4, 8, 16],
        "optimizer__nesterov" : [False, True],
    }

    rs = RandomizedSearchCV(net, params, n_iter=1, refit=True, cv=3, scoring="roc_auc", verbose = 2)

    rs.fit(X_sl, y_sl)

    print(rs.best_score_, rs.best_params_)

    #model = DenseNet121(spatial_dims=3, in_channels=8, out_channels=3)
    #
    #
    #summary(model, input_size=(5, 8, 32, 256, 256))