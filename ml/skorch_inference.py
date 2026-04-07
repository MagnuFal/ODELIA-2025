from monai.networks.nets import DenseNet121
from skorch import NeuralNetClassifier
from .dataset_class import ODELIA_SKORCH_DATASET
from skorch.helper import SliceDataset

def create_net():
    return DenseNet121(spatial_dims=3, in_channels=8, out_channels=3)

# 1️⃣ Recreate the net (same config as training)
net = NeuralNetClassifier(
    module=create_net,
    max_epochs=50,               # can match training or not important if just evaluating
    lr=1e-3,
    optimizer__momentum=0,
    device='cpu',               # use cpu
    train_split=False,
)

net.initialize()

# 2️⃣ Load the saved parameters
save_checkpoint_path = r"C:\Users\magfa\Documents\ODELIA-2025\checkpoints\skorch_run_1.pt"
net.load_params(f_params=save_checkpoint_path)

annotation_file = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\annotation.csv"
img_dir = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\RSH_np_arrays"

dataset = ODELIA_SKORCH_DATASET(annotation_file=annotation_file, img_dir=img_dir)
X_sl = SliceDataset(dataset, idx=0)
y_sl = SliceDataset(dataset, idx=1)

# 3️⃣ Use the net
preds = net.predict(X_sl)        # predictions
probs = net.predict_proba(X_sl)  # class probabilities

print(preds[0])