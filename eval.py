import torch
import clip
from tqdm import tqdm
from models.bilinearclip import BilinearCLIP
import argparse
from data_loader import get_dataset
import csv
from utils import get_config_file

from settings import *
from PIL import Image, ImageFile

from utils import get_zeroshot_weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def log_results_to_csv(dataset_name, orig_acc, bilinear_acc, filename="experiment_results.csv"):
    file_exists = os.path.isfile(filename)

    headers = ["Dataset", "Vanilla_Acc", "Bilinear_Acc", "Accuracy_Gain"]

    acc_gain = bilinear_acc - orig_acc

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow([
            dataset_name,
            f"{orig_acc:.2f}",
            f"{bilinear_acc:.2f}",
            f"{acc_gain:.2f}",

        ])

def fix_grammar(prompt):
    vowels = ('a', 'e', 'i', 'o', 'u')
    if "a photo of a " in prompt:
        object_name = prompt.split("a photo of a ")[1]
        if object_name.startswith(vowels):
            return prompt.replace("a photo of a ", "a photo of an ")
    return prompt

def evaluate_zero_shot(eval_model, dataset, preproces_model=None, is_bilinear=True):

    if preproces_model is None:
        train_loader, test_loader, prompt, classes = get_dataset(dataset, eval_model)
    else:
        train_loader, test_loader, prompt, classes = get_dataset(dataset, preproces_model)

    if dataset == "imagenet":
        if is_bilinear:
            text_features = get_zeroshot_weights(eval_model.model, classes, device)
        else:
            text_features = get_zeroshot_weights(eval_model, classes, device)
        text_features = text_features.t()
    else:
        templates = [prompt % c for c in classes]
        # templates = fix_grammar(templates)

        text_tokens = clip.tokenize(templates).to(device)

    eval_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        try:
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(device).float()
                labels = labels.to(device)

                if is_bilinear:
                    if dataset == "imagenet":
                        logits, _ = eval_model(images, text_features=text_features)
                    else:
                        logits, _ = eval_model(images, text_tokens=text_tokens)
                else:
                    if dataset == "imagenet":
                        I_f = eval_model.encode_image(images)
                        I_f = I_f/I_f.norm(dim=1, keepdim=True)
                        logits = eval_model.logit_scale.exp() * I_f @ text_features.t()

                    else:
                        logits, _ = eval_model(images, text_tokens)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        except Exception as e:
            print(e)


    return 100 * correct / total



def evaluation(config, original=False):
    Dataset = config["Dataset"]
    Model = config["Model"]
    Training = config["Training"]

    vit_model_name = Model["backbone"]

    dataset = Dataset["dataset"]
    print("Backbone:", vit_model_name)
    print("Dataset:", dataset)

    original_clip, _ = clip.load(vit_model_name, device=device)
    original_clip.float()

    upper_triangle = Model["upper_triangle"]

    model = BilinearCLIP(vit_model_name, upper_triangle=upper_triangle).to(device)
    model.float()

    num_shot = Dataset["n_shot"]
    if num_shot > 0:
        checkpoint_path = f"best_bilinear_clip_{dataset}_{num_shot}"
    else:
        checkpoint_path = f"best_bilinear_clip_{dataset}"

    if vit_model_name != "ViT-B/16":
        save_model_name = vit_model_name.replace("/", "")
        checkpoint_path = f"{save_model_name}_{checkpoint_path}"

    checkpoint_path = os.path.join(MODEL_DATA, f"{checkpoint_path}.pth")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    orig_acc = 0
    bilinear_acc = 0
    if original:
        orig_acc = evaluate_zero_shot(original_clip, dataset, model, is_bilinear=False)
    else:
        print("Skipping evaluation of zeroshot clip")
    bilinear_acc = evaluate_zero_shot(model, dataset, is_bilinear=True)

    print(f"\n--- Zero-Shot Results ---")
    print(f"Standard CLIP Accuracy:  {orig_acc:.2f}%")
    print(f"Bilinear CLIP Accuracy:  {bilinear_acc:.2f}%")
    model_name_clean = vit_model_name.replace("/", "").replace("-", "_").lower()
    if num_shot > 0:
        filename = f"c_experiment_results_{num_shot}_{model_name_clean}.csv"
    else:
        filename = "c_experiment_results.csv"
    log_results_to_csv(dataset, orig_acc, bilinear_acc, filename=filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('-n', '--num_shot', type=str, default=16,
                        help='Specify the experiment name.')
    parser.add_argument('-b', '--backbone', type=str, default="vit16",
                        help='Specify the backbone (rn50, vit16, vit32).')
    parser.add_argument('-o', '--original', action='store_true',
                        help='Run eval on clip.')

    args = parser.parse_args()

    cfg = get_config_file(args.dataset, args.num_shot, args.backbone)

    evaluation(cfg, args.original)