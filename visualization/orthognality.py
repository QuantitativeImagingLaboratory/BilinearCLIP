import torch
import os
from models.bilinearclip import BilinearCLIP
from settings import MODEL_DATA

def quantify_orthogonality(model):
    model.eval()

    with torch.no_grad():

        W = model.W

        D = W.shape[0]


        I = torch.eye(D, device=W.device)

        WTW = torch.matmul(W.t(), W)

        diff = WTW - I
        ortho_error = torch.norm(diff, p='fro')

        normalized_error = ortho_error / D

    return ortho_error.item(), normalized_error.item()


def log_results_to_csv(dataset_name, ortho_error, normalized_error, filename="orthogonality.csv"):
    file_exists = os.path.isfile(filename)

    headers = ["Dataset", "orthogonality_error", "normalized_error"]

    import csv

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow([
            dataset_name,
            f"{ortho_error:.3f}",
            f"{normalized_error:.3f}",
        ])

def orthogonality_analysis(dataset, device):
    model = BilinearCLIP("ViT-B/16", upper_triangle=True).to(device)
    model.float()

    num_shot = 16
    if num_shot > 0:
        checkpoint_path = f"best_bilinear_clip_{dataset}_{num_shot}.pth"
    else:
        checkpoint_path = f"best_bilinear_clip_{dataset}.pth"

    checkpoint_path = os.path.join(MODEL_DATA, checkpoint_path)
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    k = quantify_orthogonality(model)

    log_results_to_csv(dataset, k[0], k[1])