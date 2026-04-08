from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from skorch.callbacks import EpochScoring
from sklearn.model_selection import RandomizedSearchCV
from monai.networks.nets import DenseNet121
from torchinfo import summary
import torch
from .dataset_class import ODELIA_SKORCH_DATASET
from torch.utils.data import DataLoader
import pickle
from scipy.stats import uniform
import torch.nn as nn
import time

def create_net():
    return DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3)

if __name__ == "__main__":
    start_time = time.perf_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    annotation_file = r"/cluster/home/magnufal/TDT4265/annotation_CAM_MHA_RUMC_UKA.csv"
    img_dir = r"/cluster/home/magnufal/TDT4265/training_data"
    #annotation_file = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\annotation.csv"
    #img_dir = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\RSH_np_arrays"
    #save_checkpoint_path = r"C:\Users\magfa\Documents\ODELIA-2025\checkpoints\skorch_run_1.pth"
    save_checkpoint_path = r"/cluster/home/magnufal/TDT4265/checkpoints/skorch_run_fine_search.pt"

    dataset = ODELIA_SKORCH_DATASET(annotation_file=annotation_file, img_dir=img_dir)

    X_sl = SliceDataset(dataset, idx=0)
    y_sl = SliceDataset(dataset, idx=1)

    model = DenseNet121(spatial_dims = 3, in_channels = 8, out_channels = 3)

    net = NeuralNetClassifier(module=create_net,
                              criterion=nn.CrossEntropyLoss,
                              max_epochs = 10, # Reduced max_epochs for RandomSearch
                              lr = 1e-3, # Same opt_mom and lr as baseline model
                              optimizer__momentum = 0,
                              verbose = 1,
                              train_split=False,
                              iterator_train__num_workers=0,
                              iterator_train__pin_memory=False,
                              device = device,)

    params = {
        "lr" : uniform(0.008, 0.012),
        "optimizer__momentum" : uniform(0.7, 0.2),
        "batch_size" : [32],
        "optimizer__nesterov" : [True],
    }

    rs = RandomizedSearchCV(net, params, n_iter=10, refit=True, cv=3, scoring="roc_auc_ovr", verbose = 2, n_jobs=1)

    rs.fit(X_sl, y_sl)

    print(f"Best Score: {rs.best_score_}, Best Parameters: {rs.best_params_}")

    rs.best_estimator_.save_params(f_params=save_checkpoint_path)

    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} seconds")