from BilinearClipHead import *
import argparse
from data_loader import get_dataset
from losses import contrastive
from settings import MODEL_DATA, MODEL_DATA_SIGLIP
from utils import *


def train(config, reload=False):

    Dataset = config["Dataset"]
    Model = config["Model"]
    Training = config["Training"]

    vit_model_name = Model["backbone"]
    dataset = Dataset["dataset"]
    print(f"Backbone: {vit_model_name}")
    print(f"Training on {dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    upper_triangle = Model["upper_triangle"]

    loss_function = Training["loss"]

    model = BilinearCLIP(vit_model_name, upper_triangle=upper_triangle).to(device)
    model.float()

    num_shot = Dataset["n_shot"]
    print(f"Training on {num_shot} shots.")
    train_loader, test_loader, prompt, classes = get_dataset(dataset, model, num_shots=num_shot)

    best_acc = 0.0
    if num_shot > 0:
        save_path = f"best_bilinear_clip_{dataset}_{num_shot}.pth"
    else:
        save_path = f"best_bilinear_clip_{dataset}.pth"
    if vit_model_name != "ViT-B/16":
        save_model_name = vit_model_name.replace("/", "")
        save_path = f"{save_model_name}_{save_path}"


    save_path = os.path.join(MODEL_DATA, save_path)

    if reload:
        print(f"Loading weights from {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])


    if dataset == "imagenet":
        text_features = get_zeroshot_weights(model.model, classes, device)
        text_features = text_features.t()
    else:
        templates = [prompt % c for c in classes]
        text_tokens = clip.tokenize(templates).to(device)

    optimizer_class = get_optimizer(Training["optimizer"])
    params = get_optimizer_params(Training, model)
    if Training["optimizer"] == "sgd":
        optimizer = optimizer_class(params, momentum = 0.9)
    else:
        optimizer = optimizer_class(params)

    if reload:
        print(f"Loading optimizer state from {save_path}")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = Training["epochs"]
    save_best = Training["save_best"]
    if "lr_scheduler" in Training:
        print("Using Scheduler")
        scheduler = get_scheduler(Training, optimizer, epochs)

    for epoch in range(epochs):


        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for images, labels in pbar:
            images = images.to(device).float()

            optimizer.zero_grad()


            if dataset == "imagenet":
                logits_per_image, logits_per_text = model(images, text_features=text_features[labels])
            else:
                batch_text = text_tokens[labels].to(device)
                logits_per_image, logits_per_text = model(images, batch_text)

            # 2. Define ground truth (diagonal indices)
            # Each image at index i matches the text at index i
            ground_truth = torch.arange(len(images), device=device)

            if loss_function == "contrastive":
                loss = contrastive(logits_per_image, logits_per_text, ground_truth)

            # 4. Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "W_norm": f"{model.W.norm().item():.2f}"
            })

            train_loss += loss.item()

            loss_info = {"loss": f"{loss.item():.4f}",
                              "W_norm": f"{model.W.norm().item():.2f}",
                             }

            pbar.set_postfix(loss_info)

        if "lr_scheduler" in Training:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1} complete. Current LR: {current_lr:.6f}")
            scheduler.step()

        if not save_best:
            continue

        model.eval()
        correct = 0
        total = 0
        text_tokens_dev = text_tokens.to(device)
        with torch.no_grad():
            try:
                for images, labels in tqdm(test_loader, desc="[Validating]"):
                    images = images.to(device).float()
                    labels = labels.to(device)

                    logits_per_image, _ = model(images, text_tokens_dev)

                    preds = logits_per_image.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            except:
                pass

        current_acc = 100 * correct / total
        print(f"\nEpoch {epoch + 1} Results: Loss {train_loss / len(train_loader):.4f} | Acc: {current_acc:.2f}%")

        if current_acc > best_acc:
            best_acc = current_acc
            print(f"New Best Accuracy! Saving model to {save_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)

    if not save_best:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="A script to process a dataset file.")

    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('-n', '--num_shot', type=str, default=16,
                        help='Specify the experiment name.')
    parser.add_argument('-b', '--backbone', type=str, default="vit16",
                        help='Specify the backbone (rn50, vit16, vit32).')
    parser.add_argument('-r', '--reload', action='store_true',
                        help='Reload weights and reset the optimizer state.')

    args = parser.parse_args()

    cfg = get_config_file(args.dataset, args.num_shot, args.backbone)

    train(cfg, reload=args.reload)