import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
# ✅ FIXED: Only import what exists in your model.py
from model import UNET  
from dataset import CTDataset
from boundary_loss import get_loss_function, SegmentationMetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------------ #
# 🔧 Configuration
# ------------------------------ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/ct_scans"
TRAIN_MASK_DIR = "data/masks"
CHECKPOINT = "/content/unet_ct/unet_ct/my_checkpoint.pth.tar"

# ------------------------------ #
# 💾 Checkpoint Utilities
# ------------------------------ #
def save_checkpoint(state, filename=CHECKPOINT):
    print("✅ Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("🔁 Loading checkpoint")
    model.load_state_dict(torch.load(checkpoint)["state_dict"])

# ------------------------------ #
# 📦 Data Loader Function
# ------------------------------ #
def get_loaders(image_dir, mask_dir, batch_size, image_height, image_width, num_workers=2, pin_memory=True):
    transform = A.Compose([
        A.Resize(image_height, image_width),
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    dataset = CTDataset(image_dir, mask_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

# ------------------------------ #
# 🧠 Training Function
# ------------------------------ #
def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    model.train()
    total_loss = 0.0
    total_metrics = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0, 'focal': 0.0}
    total_dice, total_iou, total_hd = 0.0, 0.0, 0.0
    valid_hd_count = 0
    valid_batches = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        targets = torch.clamp(targets, 0, 1)

        predictions = model(data)
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"⚠️ NaN/Inf in predictions at batch {batch_idx}")
            continue

        try:
            # Handle the new loss function that returns (loss, loss_dict)
            loss, loss_dict = loss_fn(predictions, targets)
            
            # Extract individual loss components
            bce_loss = loss_dict.get('bce', torch.tensor(0.0))
            dice_loss = loss_dict.get('dice', torch.tensor(0.0))
            boundary_loss = loss_dict.get('boundary', torch.tensor(0.0))
            focal_loss = loss_dict.get('focal', torch.tensor(0.0))
                
        except Exception as e:
            print(f"❌ Error computing loss at batch {batch_idx}: {e}")
            continue

        # Skip infinite or NaN losses
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        total_loss += loss.item()
        valid_batches += 1
        
        # Accumulate individual loss components
        total_metrics['bce'] += bce_loss.item() if hasattr(bce_loss, 'item') else float(bce_loss)
        total_metrics['dice'] += dice_loss.item() if hasattr(dice_loss, 'item') else float(dice_loss)
        total_metrics['boundary'] += boundary_loss.item() if hasattr(boundary_loss, 'item') else float(boundary_loss)
        total_metrics['focal'] += focal_loss.item() if hasattr(focal_loss, 'item') else float(focal_loss)

        # ✅ FIXED: Apply sigmoid to predictions for metrics computation
        pred_sigmoid = torch.sigmoid(predictions)
        
        # ✅ Compute Dice, IoU, and Hausdorff for each image
        for i in range(predictions.shape[0]):
            dice_score = SegmentationMetrics.dice_score(pred_sigmoid[i:i+1], targets[i:i+1])
            iou_score = SegmentationMetrics.iou_score(pred_sigmoid[i:i+1], targets[i:i+1])
            hd = SegmentationMetrics.hausdorff_distance(pred_sigmoid[i, 0], targets[i, 0])
            
            total_dice += dice_score
            total_iou += iou_score
            if not np.isinf(hd) and hd < 1000:  # Filter out extremely large values
                total_hd += hd
                valid_hd_count += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loop.set_description("🧠 Training")
        loop.set_postfix(
            loss=f"{loss.item():.2f}",
            bce=f"{bce_loss.item() if hasattr(bce_loss, 'item') else bce_loss:.3f}",
            dice=f"{dice_loss.item() if hasattr(dice_loss, 'item') else dice_loss:.3f}",
            boundary=f"{boundary_loss.item() if hasattr(boundary_loss, 'item') else boundary_loss:.3f}"
        )

    # Calculate averages
    if valid_batches == 0:
        print("⚠️ No valid batches in training!")
        return float('inf'), total_metrics

    avg_loss = total_loss / valid_batches
    avg_metrics = {k: v / valid_batches for k, v in total_metrics.items()}
    
    # ✅ FIXED: Use total samples instead of dataset length for proper averaging
    total_samples = valid_batches * BATCH_SIZE
    avg_metrics.update({
        'dice_score': total_dice / total_samples,
        'iou': total_iou / total_samples,
        'hausdorff': total_hd / valid_hd_count if valid_hd_count > 0 else 0.0
    })

    return avg_loss, avg_metrics

# ------------------------------ #
# 🧪 Evaluation Function
# ------------------------------ #
@torch.no_grad()
def eval_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_metrics = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0, 'focal': 0.0}
    total_dice, total_iou, total_hd = 0.0, 0.0, 0.0
    valid_hd_count = 0
    valid_batches = 0
    
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()
        targets = torch.clamp(targets, 0, 1)
        predictions = model(data)
        
        try:
            # Handle the new loss function that returns (loss, loss_dict)
            loss, loss_dict = loss_fn(predictions, targets)
            
            # Extract individual loss components
            bce_loss = loss_dict.get('bce', torch.tensor(0.0))
            dice_loss = loss_dict.get('dice', torch.tensor(0.0))
            boundary_loss = loss_dict.get('boundary', torch.tensor(0.0))
            focal_loss = loss_dict.get('focal', torch.tensor(0.0))
            
        except Exception as e:
            print(f"❌ Error computing validation loss at batch {batch_idx}: {e}")
            continue
        
        # Skip infinite or NaN losses in validation too
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Skipping invalid validation loss: {loss.item()}")
            continue
            
        total_loss += loss.item()
        valid_batches += 1
        
        # Accumulate individual loss components
        total_metrics['bce'] += bce_loss.item() if hasattr(bce_loss, 'item') else float(bce_loss)
        total_metrics['dice'] += dice_loss.item() if hasattr(dice_loss, 'item') else float(dice_loss)
        total_metrics['boundary'] += boundary_loss.item() if hasattr(boundary_loss, 'item') else float(boundary_loss)
        total_metrics['focal'] += focal_loss.item() if hasattr(focal_loss, 'item') else float(focal_loss)

        # ✅ FIXED: Apply sigmoid to predictions for metrics computation
        pred_sigmoid = torch.sigmoid(predictions)
            
        # ✅ Compute Dice, IoU, and Hausdorff for each image
        for i in range(predictions.shape[0]):
            dice_score = SegmentationMetrics.dice_score(pred_sigmoid[i:i+1], targets[i:i+1])
            iou_score = SegmentationMetrics.iou_score(pred_sigmoid[i:i+1], targets[i:i+1])
            hd = SegmentationMetrics.hausdorff_distance(pred_sigmoid[i, 0], targets[i, 0])
            
            total_dice += dice_score
            total_iou += iou_score
            if not np.isinf(hd) and hd < 1000:  # Filter out extremely large values
                total_hd += hd
                valid_hd_count += 1
    
    # Avoid division by zero
    if valid_batches == 0:
        print("⚠️ No valid batches in validation!")
        return float('inf'), total_metrics
    
    avg_loss = total_loss / valid_batches
    avg_metrics = {k: v / valid_batches for k, v in total_metrics.items()}
    
    # ✅ FIXED: Use total samples instead of dataset length for proper averaging
    total_samples = valid_batches * BATCH_SIZE
    avg_metrics.update({
        'dice_score': total_dice / total_samples,
        'iou': total_iou / total_samples,
        'hausdorff': total_hd / valid_hd_count if valid_hd_count > 0 else 0.0
    })
    
    return avg_loss, avg_metrics

# ------------------------------ #
# 🚀 Main Function
# ------------------------------ #
def main():
    # ✅ FIXED: Only use ImprovedUNET since that's what exists
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    
    # ✅ Use the new loss function from boundary_loss.py
    loss_fn = get_loss_function()
    
    # ✅ FIXED: Reduced learning rate for better convergence
    optimizer = optim.Adam(model.parameters(), lr=LR/2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # ✅ Load training and validation data
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
        NUM_WORKERS, PIN_MEMORY
    )
    
    # ✅ Load pre-trained checkpoint if specified
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT, model)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")
        
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
        print(f"   Train - BCE: {train_metrics['bce']:.4f}, Dice: {train_metrics['dice']:.4f}, Boundary: {train_metrics['boundary']:.4f}")
        print(f"   Train - Dice Score: {train_metrics['dice_score']:.4f}, IoU: {train_metrics['iou']:.4f}, Hausdorff: {train_metrics['hausdorff']:.4f}")
        print(f"   Val   - BCE: {val_metrics['bce']:.4f}, Dice: {val_metrics['dice']:.4f}, Boundary: {val_metrics['boundary']:.4f}")
        print(f"   Val   - Dice Score: {val_metrics['dice_score']:.4f}, IoU: {val_metrics['iou']:.4f}, Hausdorff: {val_metrics['hausdorff']:.4f}")

if __name__ == "__main__":
    main()