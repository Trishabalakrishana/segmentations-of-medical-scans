#train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from model import UNET
from dataset import ChestXrayDataset
from boundary_loss import CombinedSegmentationLoss
from test import compute_metrics  # ✅ Importing from test.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/images"
TRAIN_MASK_DIR = "data/masks"
CHECKPOINT = "/content/unet_week3/unet_crf/my_checkpoint.pth.tar"

def save_checkpoint(state, filename=CHECKPOINT):
    print("✅ Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("🔁 Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint)["state_dict"])

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    total_loss = 0
    total_metrics = {'bce_loss': 0, 'dice_loss': 0, 'boundary_loss': 0}
    total_dice, total_iou, total_hd = 0, 0, 0
    valid_hd_count = 0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        targets = targets.float()
        targets = torch.clamp(targets, 0, 1)

        predictions = model(data)

        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"Warning: NaN or inf detected in predictions at batch {batch_idx}")
            continue

        loss, metrics = loss_fn(predictions, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        # Calculate metrics
        for i in range(predictions.shape[0]):
            dice, iou, hd = compute_metrics(predictions[i], targets[i])
            total_dice += dice
            total_iou += iou
            if not np.isinf(hd):
                total_hd += hd
                valid_hd_count += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loop.set_description(f"🧠 Training")
        loop.set_postfix(
            loss=loss.item(),
            bce=metrics['bce_loss'],
            dice=metrics['dice_loss'],
            boundary=metrics['boundary_loss']
        )

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in total_metrics.items()}
    avg_dice = total_dice / (len(loader.dataset))
    avg_iou = total_iou / (len(loader.dataset))
    avg_hd = total_hd / valid_hd_count if valid_hd_count > 0 else 0.0

    avg_metrics.update({'dice_score': avg_dice, 'iou': avg_iou, 'hausdorff': avg_hd})
    return avg_loss, avg_metrics

@torch.no_grad()
def eval_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    total_metrics = {'bce_loss': 0, 'dice_loss': 0, 'boundary_loss': 0}
    total_dice, total_iou, total_hd = 0, 0, 0
    valid_hd_count = 0

    for data, targets in loader:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        targets = targets.float()
        targets = torch.clamp(targets, 0, 1)

        predictions = model(data)
        loss, metrics = loss_fn(predictions, targets)

        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        # Compute segmentation metrics
        for i in range(predictions.shape[0]):
            dice, iou, hd = compute_metrics(predictions[i], targets[i])
            total_dice += dice
            total_iou += iou
            if not np.isinf(hd):
                total_hd += hd
                valid_hd_count += 1

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in total_metrics.items()}
    avg_dice = total_dice / (len(loader.dataset))
    avg_iou = total_iou / (len(loader.dataset))
    avg_hd = total_hd / valid_hd_count if valid_hd_count > 0 else 0.0

    avg_metrics.update({'dice_score': avg_dice, 'iou': avg_iou, 'hausdorff': avg_hd})
    return avg_loss, avg_metrics

def get_loaders(img_dir, mask_dir, batch_size, image_height, image_width, num_workers=2, pin_memory=True):
    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    transform = A.Compose([
        A.Resize(image_height, image_width),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = ChestXrayDataset(img_dir, mask_dir, transform=transform)
    val_ds = ChestXrayDataset(img_dir, mask_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = CombinedSegmentationLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.1,
        smooth=1e-6
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE,
        IMAGE_HEIGHT, IMAGE_WIDTH, NUM_WORKERS, PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT), model)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_metrics = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, val_metrics = eval_fn(val_loader, model, loss_fn)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            })

        print(f"✅ Train Loss: {train_loss:.4f} | 🔍 Val Loss: {val_loss:.4f}")
        print(f"   Train - BCE: {train_metrics['bce_loss']:.4f}, Dice Loss: {train_metrics['dice_loss']:.4f}, Boundary: {train_metrics['boundary_loss']:.4f}")
        print(f"   Train - Dice: {train_metrics['dice_score']:.4f}, IoU: {train_metrics['iou']:.4f}, Hausdorff: {train_metrics['hausdorff']:.4f}")
        print(f"   Val   - BCE: {val_metrics['bce_loss']:.4f}, Dice Loss: {val_metrics['dice_loss']:.4f}, Boundary: {val_metrics['boundary_loss']:.4f}")
        print(f"   Val   - Dice: {val_metrics['dice_score']:.4f}, IoU: {val_metrics['iou']:.4f}, Hausdorff: {val_metrics['hausdorff']:.4f}")

if __name__ == "__main__":
    main()
