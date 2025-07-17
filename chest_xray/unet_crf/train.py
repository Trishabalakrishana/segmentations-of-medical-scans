import os
import torch
import torch.nn as nn
import numpy as np
from utils import compute_iou, compute_hausdorff 
import torch.optim as optim
from scipy.spatial.distance import directed_hausdorff
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    CombinedLoss,
    save_predictions_as_imgs
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
from predict_and_crf import apply_crf_to_prediction  # ✅ CRF API for validation only

# ✅ Configuration
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
PIN_MEMORY = True

IMG_DIR = "data/images"
MASK_DIR = "data/masks"

if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"❌ Image directory not found: {IMG_DIR}")
if not os.path.exists(MASK_DIR):
    raise FileNotFoundError(f"❌ Mask directory not found: {MASK_DIR}")

# 🔁 Data augmentations
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=35, p=0.3),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2(),
])

# 🧲 Dice Loss (not used but retained if needed)
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# 🏋️ Training loop
def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader, desc="Training")

    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_hausdorff = 0

    all_ce, all_dice_loss, all_boundary = 0, 0, 0

    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().to(device)

        with torch.amp.autocast(device_type=device):
            outputs = model(data)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss, losses = loss_fn(logits, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_ce += losses['ce_loss']
        all_dice_loss += losses['dice_loss']
        all_boundary += losses['boundary_loss']

        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()

        total_dice += (2 * (preds * targets).sum().item()) / ((preds + targets).sum().item() + 1e-8)
        total_iou += compute_iou(preds, targets)
        total_hausdorff += compute_hausdorff(preds[0], targets[0])

        loop.set_postfix(loss=loss.item())

    num_batches = len(loader)

    print(f"Train - CE: {all_ce/num_batches:.4f}, Dice: {all_dice_loss/num_batches:.4f}, Boundary: {all_boundary/num_batches:.4f}")
    print(f"📊 Train Dice: {total_dice/num_batches:.4f} | IoU: {total_iou/num_batches:.4f} | Hausdorff: {total_hausdorff/num_batches:.2f}")

    return total_loss / num_batches


# 🧪 Validation loop with CRF refinement
def validate_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_hausdorff = 0

    all_ce, all_dice, all_boundary = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.float().to(device)

            with torch.amp.autocast(device_type=device):
                outputs = model(data)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss, losses = loss_fn(logits, targets)
                total_loss += loss.item()

            preds = torch.sigmoid(logits)

            # 🔁 Apply CRF refinement via simple function
            crf_preds = []
            for i in range(data.size(0)):
                img_np = (data[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pred_mask_np = (preds[i].squeeze().cpu().numpy() * 255).astype(np.uint8)

                refined = apply_crf_to_prediction(img_np, pred_mask_np)
                crf_tensor = torch.from_numpy(refined / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                crf_preds.append(crf_tensor)

            crf_preds = torch.cat(crf_preds, dim=0)
            bin_preds = (crf_preds > 0.5).float()

            all_ce += losses['ce_loss']
            all_dice += losses['dice_loss']
            all_boundary += losses['boundary_loss']

            total_dice += (2 * (bin_preds * targets).sum().item()) / ((bin_preds + targets).sum().item() + 1e-8)
            total_iou += compute_iou(bin_preds, targets)
            total_hausdorff += compute_hausdorff(bin_preds[0], targets[0])  # Just one sample for speed

    model.train()
    num_batches = len(loader)

    print(f"Val  - CE: {all_ce/num_batches:.4f}, Dice: {all_dice/num_batches:.4f}, Boundary: {all_boundary/num_batches:.4f}")
    print(f"📊 Val Dice (CRF): {total_dice/num_batches:.4f} | IoU: {total_iou/num_batches:.4f} | Hausdorff: {total_hausdorff/num_batches:.2f}")

    return total_loss / num_batches


# 🚀 Main training script
def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device=DEVICE)

    train_loader, val_loader, test_loader = get_loaders(
        IMG_DIR, MASK_DIR, BATCH_SIZE,
        train_transform, val_transform,
        NUM_WORKERS, PIN_MEMORY
    )

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n🧑‍🧬 Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        val_loss = validate_fn(val_loader, model, loss_fn, DEVICE)

        print(f"✅ Train Loss: {train_loss:.4f} | 🔍 Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()})
        check_accuracy(val_loader, model, DEVICE)
        save_predictions_as_imgs(val_loader, model, folder="saved_images", device=DEVICE)

    # 📊 Plotting loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # 📊 Final Evaluation on Test Set
    print("\n📊 Final Evaluation on Test Set")
    check_accuracy(test_loader, model, DEVICE)
    save_predictions_as_imgs(test_loader, model, folder="test_images", device=DEVICE)

if __name__ == "__main__":
    main()
