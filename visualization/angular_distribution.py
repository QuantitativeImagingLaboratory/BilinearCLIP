import torch
# from BilinearClipHead import BilinearCLIP
from models.bilinearclip import BilinearCLIP
import clip
from data_loader import get_dataset
import numpy as np
import os
from settings import MODEL_DATA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_congestion(zs_match, zs_unmatch, label_matcing, label_unmatching, color_m="grey", color_u="grey", title="Angular Separation: Postive vs. Negative Pairs (CLIP)", save_name=None):
    from scipy.stats import gaussian_kde
    kde_match = gaussian_kde(zs_match)
    kde_unmatch = gaussian_kde(zs_unmatch)

    x = np.linspace(min(min(zs_match), min(zs_unmatch)),
                    max(max(zs_match), max(zs_unmatch)), 1000)

    p1 = kde_match(x)
    p2 = kde_unmatch(x)
    plt.figure(figsize=(12, 7))

    sns.kdeplot(zs_match, color=color_m, linestyle="--", label=label_matcing)
    sns.kdeplot(zs_unmatch, color=color_u, alpha=0.9, label=label_unmatching)

    from scipy.integrate import simpson
    overlap_area = simpson(np.minimum(p1, p2), x)
    print(f"Manifold Overlap Area: {overlap_area:.4f}")


    plt.rcParams.update({
        'figure.autolayout': True,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'font.size': 12
    })


    plt.fill_between(x, np.minimum(p1, p2),
                     color='red',
                     alpha=0.2,
                     label=f'Overlap Area (Congestion): {overlap_area:.3f}')

    plot_y_max = np.max([np.max(p1), np.max(p2)]) * 1.05
    plt.xlim((65, 85))
    plt.ylim((0, plot_y_max))

    plt.title(title, fontsize=20)
    plt.xlabel("Angle (Degrees)", fontsize=12)
    plt.ylabel("Density")
    plt.legend(fontsize=14)
    plt.grid(alpha=0.2)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved {save_name} with size consistency.")
    plt.show()
    return overlap_area


@torch.no_grad()
def plot_matching_vs_unmatching(model, dataloader, class_names, prompt_template, dataset_name, device='cuda', num_negatives=5, samples_to_polt=-1):
    model.eval()

    zs_match = []
    zs_unmatch = []
    ad_match = []
    ad_unmatch = []

    prompts = [prompt_template % c for c in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    T_f = model.model.encode_text(text_tokens)
    T_f = T_f / T_f.norm(dim=-1, keepdim=True)

    count = 0

    samples_to_polt = samples_to_polt if samples_to_polt > 0 else len(dataloader.dataset)
    print("Samples to polt:", samples_to_polt)
    pbar = tqdm(dataloader, desc=f"Dataset: {dataset_name}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        I_f = model.model.encode_image(images)
        I_f = I_f / I_f.norm(dim=-1, keepdim=True)

        I_f_ad = (I_f @ model.W)
        I_f_ad = I_f_ad / I_f_ad.norm(dim=-1, keepdim=True)

        for i in range(len(labels)):
            gt_idx = labels[i].item()

            # --- Zero Shot ---
            sim_match = torch.dot(I_f[i], T_f[gt_idx]).clamp(-1, 1)
            zs_match.append(torch.acos(sim_match).item() * (180 / np.pi))

            # Unmatching (Sample random wrong classes)
            wrong_indices = [idx for idx in range(len(class_names)) if idx != gt_idx]
            selected_wrong = np.random.choice(wrong_indices, num_negatives, replace=False)
            for w_idx in selected_wrong:
                sim_unmatch = torch.dot(I_f[i], T_f[w_idx]).clamp(-1, 1)
                zs_unmatch.append(torch.acos(sim_unmatch).item() * (180 / np.pi))

            # --- BiCLIP ---
            # Matching
            sim_match_ad = torch.dot(I_f_ad[i], T_f[gt_idx]).clamp(-1, 1)
            ad_match.append(torch.acos(sim_match_ad).item() * (180 / np.pi))

            # Unmatching
            for w_idx in selected_wrong:
                sim_unmatch_ad = torch.dot(I_f_ad[i], T_f[w_idx]).clamp(-1, 1)
                ad_unmatch.append(torch.acos(sim_unmatch_ad).item() * (180 / np.pi))
        pbar.set_postfix({"status": f"{count}/{samples_to_polt}"})
        count += len(images)
        if count > samples_to_polt:
            break

    clip_overlap = plot_congestion(zs_match, zs_unmatch,
                    "CLIP: Positive Pairs",
                    "CLIP: Negative Pairs",
                    save_name=f"plots/{dataset_name}_angular_dist_clip.png")
    biclip_overlap = plot_congestion(ad_match, ad_unmatch,
                    "BiCLIP: Positive Pairs",
                    "BiCLIP: Negative Pairs",
                    "green",
                    "red",
                    "Angular Separation: Positive vs. Negative Pairs (BiCLIP)",
                    save_name=f"plots/{dataset_name}_angular_dist_biclip.png")

    log_results_to_csv(dataset_name, clip_overlap, biclip_overlap)

def log_results_to_csv(dataset_name, clip_dist, biclip_dist, filename="angular_distribution.csv"):
    file_exists = os.path.isfile(filename)

    headers = ["Dataset", "CLIP", "BiCLIP", "Delta"]

    delta = clip_dist - biclip_dist

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow([
            dataset_name,
            f"{clip_dist:.3f}",
            f"{biclip_dist:.3f}",
            f"{delta:.3f}",
        ])

def angular_distribution(dataset):
    original_clip, _ = clip.load("ViT-B/16", device=device)

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

    _, test_dataset, prompt, classes, labels = get_dataset(dataset.lower(), model, include_labels=True)

    plot_matching_vs_unmatching(model, test_dataset, classes, prompt, dataset.lower())