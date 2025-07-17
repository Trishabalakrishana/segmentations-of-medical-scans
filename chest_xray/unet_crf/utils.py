import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from dataset import ChestXrayDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split, Subset

# -------------------- Loss Functions --------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def compute_distance_transform(self, mask):
        batch_size = mask.shape[0]
        dt_masks = []

        for i in range(batch_size):
            if mask.dim() == 4:
                mask_np = mask[i, 0].detach().cpu().numpy()
            else:
                mask_np = mask[i].detach().cpu().numpy()

            mask_np = (mask_np > 0.5).astype(np.uint8)

            if mask_np.sum() > 0:
                dt = distance_transform_edt(1 - mask_np)
            else:
                dt = np.zeros_like(mask_np)

            dt_masks.append(torch.from_numpy(dt).float())

        if mask.dim() == 4:
            return torch.stack(dt_masks).unsqueeze(1).to(mask.device)
        else:
            return torch.stack(dt_masks).to(mask.device)

    def forward(self, pred, target):
        dt_target = self.compute_distance_transform(target)
        boundary_loss = torch.mean(pred * dt_target)
        return boundary_loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, boundary_weight=0.1, dice_weight=1.0, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.ce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, target):
        if isinstance(logits, tuple):
            logits = logits[0]  # ✅ Extract primary output if model returns tuple

        target = target.float()
        target = torch.clamp(target, 0, 1)

        ce_loss = self.ce_loss(logits, target)
        pred_probs = torch.sigmoid(logits)
        pred_probs = torch.clamp(pred_probs, self.smooth, 1 - self.smooth)
        dice_loss = self.dice_loss(pred_probs, target)
        boundary_loss = self.boundary_loss(pred_probs, target)

        total_loss = (
            self.ce_weight * ce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss
        )

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'total_loss': total_loss.item()
        }


# -------------------- Data Loader Utility --------------------

from torch.utils.data import Subset

def get_loaders(image_dir, mask_dir, batch_size,
                train_transform, val_transform,
                num_workers=4, pin_memory=True):
    
    full_dataset = ChestXrayDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,
    )

    total_len = len(full_dataset)
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)

    indices = list(range(total_len))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Apply transforms to each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("✅ Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("📦 Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

# -------------------- Accuracy Checker --------------------

def check_accuracy(loader, model, device="cuda"):
    print("🔍 Checking accuracy...")
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_dice = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if y.ndim == 3:
                y = y.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            preds = torch.sigmoid(logits)
            preds_bin = (preds > 0.5).float()

            # 🔍 Accuracy
            total_correct += (preds_bin == y).float().sum().item()
            total_pixels += y.numel()

            # 🔍 Dice Score
            intersection = (preds_bin * y).sum().item()
            union = preds_bin.sum().item() + y.sum().item()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            total_dice += dice

    accuracy = 100 * total_correct / total_pixels
    avg_dice = total_dice / len(loader)

    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"📊 Dice Score: {avg_dice:.4f}")
    model.train()



def save_predictions_as_imgs(loader, model, folder="saved_predictions/", device="cuda"):
    print("💾 Saving predictions as images...")
    model.eval()
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            for i in range(preds.shape[0]):
                save_image(preds[i], os.path.join(folder, f"pred_{idx}_{i}.png"))

    model.train()

# -------------------- Metrics --------------------

def compute_iou(preds, targets):
    intersection = (preds * targets).sum().item()
    union = ((preds + targets) > 0).sum().item()
    return intersection / (union + 1e-8)

def compute_hausdorff(pred, target):
    pred = pred.squeeze().detach().cpu().numpy()
    target = target.squeeze().detach().cpu().numpy()

    pred_points = np.argwhere(pred > 0.5)
    target_points = np.argwhere(target > 0.5)

    if pred_points.size == 0 or target_points.size == 0:
        return 0.0

    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]
    return max(hd1, hd2)
