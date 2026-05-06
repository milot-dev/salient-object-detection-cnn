"""
Purpose:
--------
Handles loading, preprocessing, augmentation, and splitting of the MSRA10K dataset
for Salient Object Detection (SOD).

Responsibilities:
-----------------
1. Load images and corresponding saliency masks
2. Apply resizing and normalization
3. Apply data augmentation to training samples
4. Provide PyTorch Dataset abstraction
5. Split dataset into train / validation / test sets
6. Return DataLoaders for the training pipeline

Dataset Notes:
--------------
- Images are .jpg files
- Masks are .png files
- Dataset pairing is done via sorted filenames (index-based pairing)
- Images are returned as RGB tensors with shape [3, H, W]
- Masks are returned as grayscale tensors with shape [1, H, W]

Important:
----------
For segmentation tasks, geometric augmentations must be applied equally
to both the image and the mask. If the image is flipped, the mask must
also be flipped in the exact same way.
"""

import os
import random
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# =========================================================
# Dataset Class
# =========================================================
class SaliencyDataset(Dataset):
    """
    Custom PyTorch Dataset for Salient Object Detection.

    Args:
        root_dir (str): Path to dataset folder containing images and masks
        img_size (int): Size to resize images and masks
        augment (bool): Whether to apply data augmentation

    Returns:
        image (Tensor): RGB image tensor [3, H, W]
        mask (Tensor): Grayscale mask tensor [1, H, W]
    """

    def __init__(self, root_dir, img_size=128, augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment

        # Load all files from directory
        files = os.listdir(root_dir)

        # Separate image and mask files
        self.image_files = sorted([f for f in files if f.endswith(".jpg")])
        self.mask_files = sorted([f for f in files if f.endswith(".png")])

        # Color augmentation is applied only to the image, not the mask
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        )

    def __len__(self):
        """Return total number of samples."""
        return len(self.image_files)

    def apply_augmentation(self, image, mask):
        """
        Apply random augmentations to both image and mask.

        Geometric transformations:
        - Horizontal flip
        - Small rotation
        - Random resized crop

        Photometric transformation:
        - Color jitter is applied only to image
        """

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation between -10 and +10 degrees
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        )

        image = TF.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=(self.img_size, self.img_size),
            interpolation=TF.InterpolationMode.BILINEAR
        )

        mask = TF.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            size=(self.img_size, self.img_size),
            interpolation=TF.InterpolationMode.NEAREST
        )

        # Color jitter only changes image appearance
        image = self.color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        """
        Retrieve image-mask pair by index.
        """

        # Construct full paths
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        mask_path = os.path.join(self.root_dir, self.mask_files[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation only for training dataset
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        else:
            image = TF.resize(image, (self.img_size, self.img_size))
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)

        # Convert to tensor and normalize to [0, 1]
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Ensure mask stays binary-like after transforms
        mask = (mask > 0.5).float()

        return image, mask


# =========================================================
# DataLoader Utility Function
# =========================================================
def get_dataloaders(data_dir, img_size=128, batch_size=16):
    """
    Creates train, validation, and test DataLoaders.

    Augmentation is applied only to the training set.
    Validation and test sets use only resizing + tensor conversion.

    Args:
        data_dir (str): Path to dataset
        img_size (int): Resize dimension
        batch_size (int): Batch size

    Returns:
        train_loader, val_loader, test_loader
    """

    # Base dataset without augmentation, used only to create consistent indices
    full_dataset = SaliencyDataset(data_dir, img_size=img_size, augment=False)

    # Compute split sizes
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Reproducible split
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create separate datasets so only training uses augmentation
    train_dataset = SaliencyDataset(data_dir, img_size=img_size, augment=True)
    val_dataset = SaliencyDataset(data_dir, img_size=img_size, augment=False)
    test_dataset = SaliencyDataset(data_dir, img_size=img_size, augment=False)

    # Apply same split indices to each dataset
    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_subset.indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader