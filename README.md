# Bilinear CLIP and SigLIP Research Project

This project explores the implementation and evaluation of **Bilinear CLIP** and **Bilinear SigLIP** models. These models extend standard Vision-Language Models (VLMs) by incorporating bilinear heads to improve performance on zero-shot and few-shot classification tasks across various datasets.

## Project Structure

The repository is organized as follows:

* **`train.py`**: The main entry point for training Bilinear CLIP models. It handles configuration loading, model initialization, and the training loop.
* **`train_siglip.py`**: Specifically designed for training Bilinear SigLIP models using the SigLIP loss function.
* **`eval.py`**: Provides functionality to evaluate the zero-shot accuracy of Bilinear CLIP models and compare them against standard CLIP.
* **`eval_siglip.py`**: Similar to `eval.py`, but tailored for evaluating Bilinear SigLIP models.
* **`data_loader.py`**: Contains the logic for loading and preprocessing various datasets, including CIFAR100, OxfordPet, Flowers102, FGVCAircraft, StanfordCars, Food101, and ImageNet.
* **`losses.py`**: Implements the loss functions used during training, such as the standard contrastive loss and the SigLIP-specific loss.
* **`utils.py`**: A collection of helper functions for seeding, optimizer and scheduler creation, and dataset-specific class name processing.
* **`settings.py`**: Manages environment-specific paths for model data and datasets.
* **`visualization.py`**: A script for generating various analytical plots, such as few-shot results, angular distributions, and orthogonality analysis.

## Installation

Code is tested using the following dependencies.

```bash
torch>=2.10.0
torchvision>=0.25.0
clip @ git+https://github.com/openai/CLIP.git@ded190a052fdf4585bd685cee5bc96e0310d2c93
open_clip_torch>=3.3.0
transformers>=5.2.0
datasets>=4.6.0
timm>=1.0.25
scikit-learn>=1.8.0
scipy>=1.17.1
opencv-python>=4.13.0.92
pillow>=12.1.1

# Data Science & Numerical Processing
numpy>=2.4.2
pandas>=3.0.1
matplotlib>=3.10.8
seaborn>=0.13.2
umap-learn>=0.5.11

# Utilities & Infrastructure
huggingface_hub>=1.4.1
PyYAML>=6.0.3
tqdm>=4.67.3
requests>=2.32.5
#pip install torch torchvision clip open_clip tqdm pyyaml pillow datasets
```

## Usage
Training
To train a Bilinear CLIP model on a specific dataset (e.g., flowers102) with 16-shot learning:

Bash
```
python train.py -d flowers102 -n 16 -b vit16
```
To train a Bilinear SigLIP model:

Bash
```
python train_siglip.py -d flowers102 -n 16 -b vit16
```
Arguments:
```
-d, --dataset: Name of the dataset.
-n, --num_shot: Number of shots for few-shot learning (default: 16).
-b, --backbone: The VLM backbone to use (e.g., vit16).
```
Evaluation
To evaluate the zero-shot performance of a trained model:

Bash
```
python eval.py -d flowers102 -n 16 -b vit16
```
Visualization
To generate visualizations for the experiments:

Bash
```
python visualization.py --few-shot --angular-dist --orthogonality
```
Supported Datasets
The project currently supports the following datasets through data_loader.py:
```
DTD
OxfordIIITPet 
Flowers102
FGVCAircraft
StanfordCars
Food101
SUN397 # Requires external download
EuroSAT
Caltech101
ImageNet # Requires external download
UCF101 # Requires external download
```