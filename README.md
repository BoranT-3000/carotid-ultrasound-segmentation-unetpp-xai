# Carotid Ultrasound Segmentation with U-Net++ and XAI

An end-to-end deep learning workflow for **common carotid artery ultrasound segmentation** using **U-Net++ (ResNet34 encoder)**, with post-processing, uncertainty estimation (MC Dropout), and Grad-CAM based explainability.

## Project Overview

This repository contains a full notebook pipeline that:

- downloads and prepares the dataset from Kaggle,
- performs EDA and augmentations,
- trains a U-Net++ segmentation model,
- evaluates with multiple segmentation metrics,
- visualizes predictions and uncertainty maps,
- generates Grad-CAM explainability outputs,
- saves training history, final metrics, and checkpoints.

## Model and Training Setup

- Architecture: `UnetPlusPlus`
- Encoder: `resnet34` (`imagenet` weights)
- Input size: `256x256`
- Batch size: `8`
- Train/Validation split: `80/20`
- Dataset size: `1100` images (`880` train, `220` validation)
- Loss: weighted combination of Dice, Focal, Tversky, Boundary losses

## Best Results (Validation)

From the saved outputs (`outputs/final_metrics.csv` and notebook summary):

- Dice: **0.9523** (best epoch Dice reached **0.9564**)
- IoU: **0.9095** (best epoch IoU reached **0.9155**)
- F1: **0.9525**
- Precision: **0.9702**
- Recall: **0.9358**
- Specificity: **0.9993**
- MCC: **0.9516**
- Balanced Accuracy: **0.9676**
- Hausdorff Distance: **3.46** (best epoch value **3.07**)

## Repository Structure

```text
.
├── carotid_segmentation_unet++.ipynb
├── checkpoints/
│   ├── best_model.pth
│   └── final_model.pth
├── outputs/
│   ├── augmentations.png
│   ├── eda_samples.png
│   ├── final_metrics.csv
│   ├── gradcam_xai.png
│   ├── predictions.png
│   ├── training_history.csv
│   └── training_history.png
└── Explainable Deep Learning for Carotid Artery Ultrasound Segmentation Using U-Net++.docx
```

## Installation

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install albumentations segmentation-models-pytorch torchmetrics
pip install grad-cam opencv-python-headless scikit-image timm kagglehub
```

## Dataset

The notebook uses:

- Kaggle dataset: `orvile/carotid-ultrasound-images`

It is downloaded in-notebook with `kagglehub`.

## How to Run

1. Open `carotid_segmentation_unet++.ipynb`.
2. Run cells in order from top to bottom.
3. Check generated artifacts in `outputs/` and `checkpoints/`.

## Notes

- The notebook includes post-processing and uncertainty visualization utilities.
- Grad-CAM outputs are saved to `outputs/gradcam_xai.png`.
- Pre-generated artifacts are already included in this repository.

