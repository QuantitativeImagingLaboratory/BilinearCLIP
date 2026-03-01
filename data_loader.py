import os
import torch

from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100, OxfordIIITPet, FGVCAircraft, Flowers102, ImageNet
from torchvision.datasets import (
    StanfordCars, Food101, SUN397, DTD,
    EuroSAT, Caltech101
)
from utils import get_flower_names, get_imagenet_classes
from torchvision import transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset
from torch.utils.data import Dataset

def get_dataset(dataset_name, model, batch_size=64, include_labels=False, num_shots=-1):
    if os.getenv("SABINE", False):
        root = os.path.expanduser(".cache")
    else:
        root = os.path.expanduser("~/.cache")

    preprocess = model.preprocess
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    class ApplyTransform(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    prompt = "a photo of a %s."
    labels = None

    if dataset_name == "CIFAR100".lower():
        train_ds = CIFAR100(root, train=True, download=True, transform=preprocess)
        test_ds = CIFAR100(root, train=False, download=True, transform=preprocess)
        classes = train_ds.classes
        labels = test_ds.targets

    elif dataset_name == "OxfordPet".lower():
        train_ds = OxfordIIITPet(root, split="trainval", download=True, transform=train_preprocess)
        test_ds = OxfordIIITPet(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        labels = test_ds._labels
        prompt = "a photo of a %s, a type of pet."

    elif dataset_name == "Flowers102".lower():
        train_ds_ = Flowers102(root, split="train", download=True, transform=train_preprocess)
        val_ds = Flowers102(root, split="val", download=True, transform=train_preprocess)

        train_ds = torch.utils.data.ConcatDataset([train_ds_, val_ds])
        test_ds = Flowers102(root, split="test", download=True, transform=preprocess)

        classes = [f"{get_flower_names(i)}" for i in range(102)]
        labels = test_ds._labels
        prompt = "a photo of a %s, a type of flower."

    elif dataset_name == "Aircraft".lower():
        train_ds = FGVCAircraft(root, split="trainval", download=True, transform=train_preprocess)
        test_ds = FGVCAircraft(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        labels = test_ds._labels

        prompt = "a photo of an %s, a type of aircraft."
    elif dataset_name == "stanfordcars":

        train_ds = StanfordCars(root, split="train", download=False, transform=train_preprocess)
        test_ds = StanfordCars(root, split="test", download=False, transform=preprocess)
        classes = train_ds.classes

    elif dataset_name == "food101":
        train_ds = Food101(root, split="train", download=True, transform=train_preprocess)
        test_ds = Food101(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        prompt = "a photo of %s, a type of food."

    elif dataset_name == "sun397":

        # full_ds = SUN397(root, download=True, transform=preprocess)
        # train_size = int(0.8 * len(full_ds))
        # test_size = len(full_ds) - train_size
        # train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])
        # classes = full_ds.classes
        # prompt = "a photo of a %s, a type of scene."

        # dataset = load_dataset("tanganke/sun397", cache_dir=root, split="train")
        dataset = load_dataset("1aurent/SUN397", split="train")

        def transform_fn(examples):
            images = [preprocess(img.convert("RGB")) for img in examples["image"]]
            labels = examples["label"]

            return {"pixel_values": images, "label": labels}

        dataset = dataset.with_transform(transform_fn)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

        classes = dataset.features["label"].names
        prompt = "a centered photo of %s."

    elif dataset_name == "dtd":
        train_ds = DTD(root, split="train", download=True, transform=train_preprocess)
        test_ds = DTD(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        prompt = "a photo of a %s texture."

    elif dataset_name == "eurosat":
        full_ds = EuroSAT(root, download=True, transform=None)

        train_size = int(0.8 * len(full_ds))
        test_size = len(full_ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

        train_ds = ApplyTransform(train_ds, transform=train_preprocess)
        test_ds = ApplyTransform(test_ds, transform=preprocess)

        classes = full_ds.classes
        prompt = "a centered satellite photo of %s."


    elif dataset_name == "caltech101":
        full_ds = Caltech101(root, download=True, transform=None)
        train_size = int(0.8 * len(full_ds))
        test_size = len(full_ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

        train_ds = ApplyTransform(train_ds, transform=train_preprocess)
        test_ds = ApplyTransform(test_ds, transform=preprocess)

        classes = full_ds.categories
        prompt = "a photo of a %s."
    elif dataset_name == "ImageNet".lower():
        def read_imagenet_classes():
            filename = "/home/pmantini/Documents/Research/clip/imagenet/classes"
            classes_loaded = []
            with open(filename, "r") as f:
                all_classes = f.readlines()

            classes_loaded += [this_classes.split(",", 1)[1] for this_classes in all_classes]

            return classes_loaded

        if os.getenv("SABINE", False):
            train_root = "imagenet/partial16"
        else:
            train_root = "/home/pmantini/Documents/Research/clip/imagenet/partial16"


        train_ds = ImageFolder(
            root=train_root,
            transform=train_preprocess  # Use the CLIP preprocess function
        )

        # train_ds = ImageNet(root=root, split="train", download=False, transform=preprocess)
        test_ds = ImageNet(root=root, split="val", transform=preprocess)
        # url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        # class_idx = requests.get(url).json()
        # all_classes = read_imagenet_classes()
        all_classes = get_imagenet_classes()
        classes = [label.replace('\n', '').strip() for label in all_classes]

        prompt = "a photo of a %s"

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    print("Processing for n-shot classification.")
    if num_shots > 0:

        if hasattr(train_ds, 'targets'):
            labels = np.array(train_ds.targets)
        elif hasattr(train_ds, '_labels'):
            labels = np.array(train_ds._labels)
        else:
            # labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
            first_item = train_ds[0]
            if isinstance(first_item, dict):
                # Extract 'label' from every dict in the dataset
                labels = np.array([train_ds[i]['label'] for i in range(len(train_ds))])
            else:
                # Fallback for standard (image, label) tuples
                labels = np.array([train_ds[i][1] for i in range(len(train_ds))])

        few_shot_indices = []
        unique_classes = np.unique(labels)

        for c in unique_classes:
            indices = np.where(labels == c)[0]

            replace = False
            if dataset_name == 'flowers102':
                replace = True
            sampled = np.random.choice(indices, num_shots, replace=replace)
            few_shot_indices.extend(sampled)


        train_ds = Subset(train_ds, few_shot_indices)
        print(f"Created {num_shots}-shot dataset with {len(train_ds)} total images.")

    clean_classes = [c.replace("_", " ").replace("-", " ").lower() for c in classes]

    if dataset_name == 'imagenet':
        batch_size = 256

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if include_labels:
        return train_loader, test_loader, prompt, clean_classes, labels
    else:
        return train_loader, test_loader, prompt, clean_classes