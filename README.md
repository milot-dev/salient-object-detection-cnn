# Salient Object Detection using CNN Encoder-Decoder

This project implements an end-to-end **Salient Object Detection (SOD)** system from scratch using **PyTorch**.  
The goal is to identify the most visually important object or region in an image and generate a one-channel saliency mask.

The model is trained on the **MSRA10K** dataset and evaluated using common segmentation metrics such as **IoU, Precision, Recall, and F1-score**.

---

## Project Overview

Salient Object Detection focuses on detecting the dominant visual region in an image.  
Unlike traditional object detection, which identifies and classifies multiple objects, SOD produces a pixel-level mask that highlights the most important object or area.

This project includes:

- Dataset loading and preprocessing
- Train / validation / test splitting
- Data augmentation
- CNN encoder-decoder model implementation from scratch
- Custom loss function
- Full training and validation loop
- Model checkpoint saving
- Evaluation metrics
- Prediction visualizations
- Upload-image demo notebook

---

## Dataset

The project uses the **MSRA10K** salient object detection dataset.

Dataset structure expected by the notebook:

```text
/content/data/MSRA10K_Imgs_GT/Imgs
```

The dataset contains:

```text
Images: 10000
Masks: 10000
```

The notebook includes an automatic dataset check and download step if the dataset is missing.

---

## Model Architecture

Two models are implemented:

### 1. BaselineSODModel

A simple CNN encoder-decoder architecture built from scratch.

Main structure:

```text
Input RGB image: 3 x 128 x 128

Encoder:
Conv2D + ReLU + MaxPooling
Conv2D + ReLU + MaxPooling
Conv2D + ReLU + MaxPooling

Decoder:
ConvTranspose2D + ReLU
ConvTranspose2D + ReLU
ConvTranspose2D + ReLU

Output:
1-channel sigmoid saliency mask
```

### 2. ImprovedSODModel

An improved CNN encoder-decoder model with:

- Deeper convolutional layers
- Batch normalization
- Dropout
- Stronger training configuration
- Data augmentation support

---

## Data Preprocessing

The dataset pipeline includes:

- Image and mask loading
- Resizing to `128 x 128`
- Normalization to `[0, 1]`
- Conversion to PyTorch tensors
- Train / validation / test split:
  - 70% training
  - 15% validation
  - 15% testing

---

## Data Augmentation

The training dataset uses basic augmentations, including:

- Horizontal flip
- Random rotation
- Random crop
- Brightness variation / color jitter

The same transformations are applied carefully so that image-mask alignment is preserved.

---

## Loss Function

The model uses a combined Binary Cross-Entropy and IoU-based loss:

```text
Total Loss = BCE + 0.5 * (1 - IoU)
```

This helps the model optimize both pixel-level accuracy and mask overlap quality.

---

## Optimizer and Training

The optimizer used is:

```text
Adam, learning rate = 1e-3
```

Both baseline and improved models were trained for up to 20 epochs.

The training loop includes:

- Forward pass
- Loss computation
- Backward pass
- Optimizer update
- Validation after each epoch
- Best model checkpoint saving
- Early stopping logic

---

## Final Results

| Model | IoU | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline Model | 0.6004 | 0.7186 | 0.8018 | 0.7247 |
| Improved Model + Augmentation | 0.7270 | 0.8068 | 0.8851 | 0.8256 |

The improved model performed better across all evaluation metrics.

### Improvement Summary

| Metric | Improvement |
|---|---:|
| IoU | +0.1266 |
| Precision | +0.0882 |
| Recall | +0.0833 |
| F1-score | +0.1009 |

---

## Project Structure

```text
SOD_Project/
│
├── data_loader.py
├── sod_model.py
├── train.py
├── evaluate.py
├── demo_notebook.ipynb
├── README.md
│
├── checkpoints/
│   ├── baseline_20ep_best.pth
│   └── improved_augmented_20ep_best.pth
├── report/
│   ├── Final_Report.pdf

```

---

## File Descriptions

### `data_loader.py`

Contains the dataset class and DataLoader creation logic.

Main responsibilities:

- Load image-mask pairs
- Resize images and masks
- Normalize values
- Apply augmentation
- Create train, validation, and test DataLoaders

---

### `sod_model.py`

Contains the CNN model architectures.

Implemented models:

- `BaselineSODModel`
- `ImprovedSODModel`

Both models are built from scratch using PyTorch layers.

---

### `train.py`

Contains the full training and validation loop.

Main features:

- Custom loss computation
- Training loop
- Validation loop
- Checkpoint saving
- Best model saving
- Early stopping logic

---

### `evaluate.py`

Contains evaluation metric logic and visualizations.

Metrics included:

- Intersection-over-Union
- Precision
- Recall
- F1-score

---

### `demo_notebook.ipynb`

The main notebook used for:

- Dataset setup
- DataLoader testing
- Model training
- Model evaluation
- Prediction visualization
- Final upload-image demo

The demo allows the user to upload any image and displays:

- Input image
- Predicted saliency mask
- Overlay visualization
- Inference time per image

---

## How to Run the Project

### 1. Open the notebook in Google Colab

Open:

```text
demo_notebook.ipynb
```

---

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 3. Set the project directory

```python
%cd /content/drive/MyDrive/SOD_Project
```

---

### 4. Install / download dataset if needed

The notebook contains an automatic dataset download cell using Kaggle.

If the dataset is missing, it downloads and extracts MSRA10K automatically.

---

### 5. Create DataLoaders

```python
from data_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    DATA_DIR,
    img_size=128,
    batch_size=16
)
```

---

### 6. Train the baseline model

```python
from sod_model import BaselineSODModel
from train import train_model
import torch.optim as optim

baseline_model = BaselineSODModel().to(device)

optimizer = optim.Adam(
    baseline_model.parameters(),
    lr=1e-3
)

train_losses, val_losses = train_model(
    model=baseline_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=20,
    checkpoint_dir="/content/drive/MyDrive/SOD_Project/checkpoints",
    model_name="baseline_20ep"
)
```

---

### 7. Train the improved model

```python
from sod_model import ImprovedSODModel

improved_model = ImprovedSODModel().to(device)

optimizer = optim.Adam(
    improved_model.parameters(),
    lr=1e-3
)

train_losses_aug, val_losses_aug = train_model(
    model=improved_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=20,
    checkpoint_dir="/content/drive/MyDrive/SOD_Project/checkpoints",
    model_name="improved_augmented_20ep"
)
```

---

### 8. Evaluate a trained model

```python
from evaluate import evaluate_model
from sod_model import ImprovedSODModel
import torch

model = ImprovedSODModel().to(device)

model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/SOD_Project/checkpoints/improved_augmented_20ep_best.pth",
        map_location=device
    )
)

metrics = evaluate_model(model, test_loader, device)

print(metrics)
```

---

## Bonus Feature: Checkpoint Save and Resume

The training loop supports automatic checkpoint saving and resume functionality.

After each epoch, the project saves:

- model weights
- optimizer state
- current epoch
- training loss history
- validation loss history
- best validation loss

If training is interrupted, rerunning the notebook automatically loads the latest checkpoint and continues from the next epoch.

Example console output:

```text
Checkpoint found: checkpoints/improved_augmented_20ep_last_checkpoint.pth
Loading checkpoint and resuming training...
Resumed training from epoch 21.
Best validation loss so far: 0.5201
Training finished.
```

## Demo Usage

The notebook includes a final demo section:

```text
Section 15: Demo — Upload Any Image and Predict Saliency Mask
```

The demo:

1. Loads the best improved model checkpoint.
2. Allows the user to upload an image.
3. Resizes the image to `128 x 128`.
4. Predicts the saliency mask.
5. Displays:
   - Input image
   - Predicted mask
   - Overlay
   - Inference time

---

## Example Demo Output

The output contains three visualizations:

```text
Input Image | Predicted Saliency Mask | Overlay
```

The overlay shows the predicted salient region on top of the original image.

---

## Requirements

Recommended environment:

```text
Python 3.9+
PyTorch
torchvision
NumPy
OpenCV
Matplotlib
Pandas
scikit-learn
tqdm
Pillow
Google Colab
```

Install common dependencies:

```bash
pip install torch torchvision numpy opencv-python matplotlib pandas scikit-learn tqdm pillow
```

---

## Dataset Download Note

The notebook automatically downloads the MSRA10K dataset from Kaggle if the dataset folder is missing.

In Google Colab, the download usually works directly without additional setup. If the download fails in another environment, manually download the dataset from Kaggle and place it in the expected folder structure:

```text
/content/data/MSRA10K_Imgs_GT/Imgs
```

This folder should contain both:
- .jpg image files
- .png mask files

## Notes

- The model is trained from scratch.
- No pretrained models are used.
- The improved model gives better results than the baseline model.
- The best checkpoint should be used for evaluation and demo.
- If training is interrupted, rerunning the notebook or training script automatically loads the latest checkpoint.

---

## Final Conclusion

This project successfully implements a complete Salient Object Detection pipeline using a CNN encoder-decoder architecture.  
The improved model achieved stronger performance than the baseline model by using a deeper architecture, batch normalization, dropout, and data augmentation.

The final system can process an input image and produce a saliency mask with overlay visualization.
