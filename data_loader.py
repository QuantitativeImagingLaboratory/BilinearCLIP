import os
import torch
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100, OxfordIIITPet, FGVCAircraft, Flowers102, ImageNet
from torchvision.datasets import (
    StanfordCars, Food101, SUN397, DTD,
    EuroSAT, Caltech101,
)
from utils import get_flower_names, get_imagenet_classes, get_eurosat_classes
from torchvision import transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
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

            if hasattr(x, 'convert'):
                x = x.convert("RGB")

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
        all_classes = train_ds.classes

        def process_stanford_cars_classes(classname):
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            return ' '.join(names)

        classes = [process_stanford_cars_classes(c) for c in all_classes]

        prompt = "a photo of %s, a type of car."

    elif dataset_name == "food101":
        train_ds = Food101(root, split="train", download=True, transform=train_preprocess)
        test_ds = Food101(root, split="test", download=True, transform=preprocess)
        classes = train_ds.classes
        prompt = "a photo of %s, a type of food."

    elif dataset_name == "sun397":
        from datasets import load_from_disk

        # Point to the folder contianing the dataset
        small_path = None
        assert small_path is not None, f"{dataset_name} Dataset is not defined. Requires manual download."

        small_bundle = load_from_disk(small_path)

        def process_class_name(raw_name):
            names = raw_name.split('/')[2:]
            names = names[::-1]
            classname = ' '.join(names)
            return classname

        classes_all = small_bundle['train'].features["label"].names
        classes = [process_class_name(c) for c in classes_all]


        class HFDatasetWrapper(Dataset):
            def __init__(self, hf_dataset, transform=None):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self._labels = self.hf_dataset['label']

            def __getitem__(self, index):
                item = self.hf_dataset[index]
                x, y = item['image'], item['label']
                if hasattr(x, 'convert'): x = x.convert("RGB")
                if self.transform: x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.hf_dataset)

        train_ds = HFDatasetWrapper(small_bundle['train'], transform=train_preprocess)
        test_ds = HFDatasetWrapper(small_bundle['test'], transform=preprocess)

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
        classes = get_eurosat_classes(classes)
        prompt = "a centered satellite photo of %s."


    elif dataset_name == "caltech101":
        full_ds = Caltech101(root, download=True, transform=None)

        def get_few_shot_split(full_ds, n_shots=16):
            indices = np.arange(len(full_ds))
            labels = [full_ds[i][1] for i in indices]

            train_indices = []
            test_indices = []

            for label in range(len(full_ds.categories)):
                label_indices = [i for i, l in enumerate(labels) if l == label]
                # Sample exactly 16 for training, everything else for testing
                np.random.shuffle(label_indices)
                train_indices.extend(label_indices[:n_shots])
                n_shot_for_test_set = 16
                test_indices.extend(label_indices[n_shot_for_test_set:n_shot_for_test_set+19])

            return Subset(full_ds, train_indices), Subset(full_ds, test_indices)
        train_ds, test_ds = get_few_shot_split(full_ds, n_shots=num_shots)

        train_ds = ApplyTransform(train_ds, transform=train_preprocess)
        test_ds = ApplyTransform(test_ds, transform=preprocess)

        classes = full_ds.categories

        prompt = "a centered photo of a %s."

    elif dataset_name == "ucf101":
        d_path = None
        s_path = None

        assert d_path is not None, f"{dataset_name} path is not defined"
        assert s_path is not None, f"{dataset_name} path is not defined"

        data_path = Path(d_path)
        split_dir = Path(s_path)


        train_list_file = split_dir / "trainlist01.txt"
        test_list_file = split_dir / "testlist01.txt"

        def load_split(list_file, is_test=False):
            samples = []
            with open(list_file, 'r') as f:
                for line in f:                    
                    parts = line.strip().split(' ')
                    video_rel_path = parts[0]
                    
                    img_rel_path = video_rel_path.replace('.avi', '.jpg')
                    img_full_path = data_path / img_rel_path
                    
                    class_name = video_rel_path.split('/')[0]

                    samples.append((str(img_full_path), class_name))
            return samples

        train_samples = load_split(train_list_file)
        test_samples = load_split(test_list_file, is_test=True)
        
        all_classes = sorted(list(set([s[1] for s in train_samples + test_samples])))
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}


        class FileListDataset(Dataset):
            def __init__(self, samples, class_to_idx, transform=None):
                self.samples = samples
                self.class_to_idx = class_to_idx
                self.transform = transform

            def __len__(self): return len(self.samples)

            def __getitem__(self, i):
                path, cls_name = self.samples[i]
                from PIL import Image
                img = Image.open(path).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, self.class_to_idx[cls_name]

        train_ds = FileListDataset(train_samples, class_to_idx, transform=train_preprocess)
        test_ds = FileListDataset(test_samples, class_to_idx, transform=preprocess)

        classes = all_classes
        import re
        classes = [re.sub(r'(?<!^)(?=[A-Z])', ' ', c) for c in classes]
        prompt = "a photo of a person %s."
    elif dataset_name == "ImageNet".lower():
        # def read_imagenet_classes():
        #     filename = "/home/pmantini/Documents/Research/clip/imagenet/classes"
        #     classes_loaded = []
        #     with open(filename, "r") as f:
        #         all_classes = f.readlines()
        #
        #     classes_loaded += [this_classes.split(",", 1)[1] for this_classes in all_classes]
        #
        #     return classes_loaded

        if os.getenv("SABINE", False):
            train_root = "imagenet/partial16"
        else:
            train_root = None

        assert train_root is not None, f"{dataset_name} path is not defined"

        train_ds = ImageFolder(
            root=train_root,
            transform=train_preprocess  # Use the CLIP preprocess function
        )


        test_ds = ImageNet(root=root, split="val", transform=preprocess)
        all_classes = get_imagenet_classes()
        classes = [label.replace('\n', '').strip() for label in all_classes]

        prompt = "a photo of a %s"

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    print(f"Processing for {num_shots}-shot classification.")
    if num_shots > 0:

        if hasattr(train_ds, 'targets'):
            labels = np.array(train_ds.targets)
        elif hasattr(train_ds, '_labels'):
            labels = np.array(train_ds._labels)
        else:
            first_item = train_ds[0]
            if isinstance(first_item, dict):
                labels = np.array([train_ds[i]['label'] for i in range(len(train_ds))])
            else:

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