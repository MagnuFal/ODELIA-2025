import torch
from torch.utils.data import DataLoader, random_split
from ml.dataset_class import ODELIA_DATASET
import torch
from monai.networks.nets import DenseNet121, DenseNet264
from data_analysis.data_analysis import txt_to_csv
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(model, dataloader, save_path):
    model.eval()
    inference_preds = rf"{save_path}\inference_probs.txt"
    #with open(inference_preds, "w") as f:
    #    f.write(f"ID, normal, benign, malignant\n")
    lst = []
    with torch.no_grad():
        for X, y, uid in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            probs = torch.softmax(pred, dim = 1)
            #with open(inference_preds, "a") as f:
            #    for i in range(len(uid)):
            #        f.write(f"{uid[i]}, {probs[i,0]}, {probs[i,1]}, {probs[i,2]}\n")
            for i in range(len(uid)):
                lst.append({"ID" : uid[i],
                            "normal" : float(probs[i,0]),
                            "benign" : float(probs[i,1]),
                            "malignant" : float(probs[i,2])},)

    #txt_to_csv(rf"{save_path}\inference_probs.txt", rf"{save_path}\inference.csv")
    df = pd.DataFrame(lst)
    df.to_csv(rf"{save_path}\new_data_representation_and_paper_training_params.csv", index = False)

if __name__ == "__main__":
    annotation_file = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\annotation.csv"
    img_dir = r"C:\Users\magfa\Documents\ODELIA-2025\RSH_dataset\RSH_np_arrays_reshaped"

    dataset = ODELIA_DATASET(annotation_file=annotation_file, img_dir=img_dir)

    test_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = 3, pretrained=False).to(device)

    checkpoint = torch.load(r"C:\Users\magfa\Documents\ODELIA-2025\checkpoints\new_data_representation_and_paper_training_params.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    inference(model=model, dataloader=test_loader, save_path=r"C:\Users\magfa\Documents\ODELIA-2025\RSH_Inference")