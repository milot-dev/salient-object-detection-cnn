"""
Purpose:
--------
Contains the training and validation logic for Salient Object Detection.

Main responsibilities:
----------------------
1. Define BCE + IoU loss
2. Train the model for one epoch
3. Validate the model
4. Save best model checkpoint
5. Save last checkpoint for resume training
"""

import os
import torch
import torch.nn as nn


# =========================================================
# IoU Loss
# =========================================================
def iou_loss(preds, targets, smooth=1e-6):
    """
    Computes IoU loss.

    IoU = intersection / union
    IoU loss = 1 - IoU

    Args:
        preds (Tensor): predicted masks [B, 1, H, W]
        targets (Tensor): ground truth masks [B, 1, H, W]

    Returns:
        Tensor: IoU loss
    """

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return 1 - iou.mean()


# =========================================================
# Combined Loss: BCE + IoU
# =========================================================
def combined_loss(preds, targets):
    """
    Combines Binary Cross Entropy loss with IoU loss.
    BCE helps pixel-level classification.
    IoU helps mask overlap quality.
    """

    bce = nn.BCELoss()(preds, targets)
    iou = iou_loss(preds, targets)

    return bce + 0.5 * iou


# =========================================================
# Train One Epoch
# =========================================================
def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains model for one epoch.
    """

    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(images)
        loss = combined_loss(preds, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# =========================================================
# Validate One Epoch
# =========================================================
def validate_one_epoch(model, dataloader, device):
    """
    Evaluates model on validation data.
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = combined_loss(preds, masks)

            total_loss += loss.item()

    return total_loss / len(dataloader)


# =========================================================
# Full Training Loop
# =========================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=20,
    checkpoint_dir="checkpoints",
    model_name="baseline_model",
    resume = True
):
    """
    Full training loop.

    Saves:
    ------
    1. best model based on validation loss
    2. last checkpoint after each epoch for resume training

    Resume logic:
    -------------
    If resume=True and a last checkpoint exists, training automatically
    continues from the saved epoch.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    start_epoch = 0

    last_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_last_checkpoint.pth"
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_best.pth"
    )

    # =====================================================
    # Auto-resume from last checkpoint
    # =====================================================
    if resume and os.path.exists(last_checkpoint_path):
        print(f"Checkpoint found: {last_checkpoint_path}")
        print("Loading checkpoint and resuming training...")

        checkpoint = torch.load(last_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

        if len(val_losses) > 0:
            best_val_loss = min(val_losses)

        print(f"Resumed training from epoch {start_epoch + 1}.")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    else:
        print("No checkpoint found. Starting training from scratch.")

    # =====================================================
    # Training loop
    # =====================================================
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # =================================================
        # Save last checkpoint after every epoch
        # =================================================
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss
        }, last_checkpoint_path)

        print(f"Last checkpoint saved: {last_checkpoint_path}")

        # =================================================
        # Save best model
        # =================================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)

            print(f"Best model saved: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        # =================================================
        # Early stopping
        # =================================================
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print("Training finished.")

    return train_losses, val_losses