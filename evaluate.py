"""
Purpose:
--------
Evaluate trained SOD model using key metrics:

- IoU (Intersection over Union)
- Precision
- Recall
- F1 Score

Also provides visualization utilities.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# =========================================================
# Metrics
# =========================================================
def compute_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Computes IoU, Precision, Recall, F1 score.

    Args:
        preds: predicted masks [B,1,H,W]
        targets: ground truth masks
    """

    preds = (preds > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)

    return iou.mean().item(), precision.mean().item(), recall.mean().item(), f1.mean().item()


# =========================================================
# Evaluate model
# =========================================================
def evaluate_model(model, dataloader, device):
    """
    Runs model on test set and computes average metrics.
    """

    model.eval()

    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            iou, precision, recall, f1 = compute_metrics(preds, masks)

            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

    return {
        "IoU": total_iou / count,
        "Precision": total_precision / count,
        "Recall": total_recall / count,
        "F1": total_f1 / count
    }

  # =========================================================
# Visualization Function
# =========================================================
def visualize_predictions(model, dataloader, device, num_samples=3, threshold=0.5):
    """
    Displays input image, ground-truth mask, predicted mask,
    and overlay visualization.

    Args:
        model: trained SOD model
        dataloader: test or validation dataloader
        device: CPU or CUDA
        num_samples: number of samples to visualize
        threshold: mask threshold for binary prediction
    """

    model.eval()
    shown = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            start_time = time.time()
            preds = model(images)
            end_time = time.time()

            inference_time = (end_time - start_time) / images.size(0)

            preds_binary = (preds > threshold).float()

            for i in range(images.size(0)):
                if shown >= num_samples:
                    return

                image = images[i].cpu().permute(1, 2, 0).numpy()
                gt_mask = masks[i].cpu().squeeze().numpy()
                pred_mask = preds_binary[i].cpu().squeeze().numpy()

                overlay = image.copy()
                overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_mask)

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 4, 1)
                plt.imshow(image)
                plt.title("Input Image")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(gt_mask, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(pred_mask, cmap="gray")
                plt.title("Predicted Mask")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(overlay)
                plt.title("Overlay")
                plt.axis("off")

                plt.suptitle(f"Inference time: {inference_time:.4f} sec/image")
                plt.tight_layout()
                plt.show()

                shown += 1