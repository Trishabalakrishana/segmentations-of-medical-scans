import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
from dataset import CTDataset  # use CTScanDataset instead of ChestXrayDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
# ------------------------------
# Dice Loss
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        intersection = (pred * target).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()
        return dice_loss

# ------------------------------
# Boundary Loss
# ------------------------------
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def compute_distance_transform(self, mask):
        batch_size = mask.shape[0]
        dt_masks = []
        for i in range(batch_size):
            mask_np = mask[i, 0].detach().cpu().numpy()
            mask_np = (mask_np > 0.5).astype(np.uint8)
            
            if mask_np.sum() > 0:
                dt = distance_transform_edt(1 - mask_np)
            else:
                dt = np.zeros_like(mask_np)
            
            dt_masks.append(torch.from_numpy(dt).float())
        
        return torch.stack(dt_masks).unsqueeze(1).to(mask.device)
    def forward(self, pred, target):
        dt_target = self.compute_distance_transform(target)
        boundary_loss = torch.mean(pred * dt_target)
        return boundary_loss
# ------------------------------
# Combined Loss
# ------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, boundary_weight=0.1, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()
        self.dice_loss = DiceLoss()  # uses default smooth inside DiceLoss
    def forward(self, logits, target):
        target = target.float()
        target = torch.clamp(target, 0, 1)
        ce_loss = self.ce_loss(logits, target)
        pred_probs = torch.sigmoid(logits)
        pred_probs = torch.clamp(pred_probs, 1e-6, 1 - 1e-6)  # prevent instability
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

# ------------------------------
# Data Loader for CT scans
# ------------------------------
def get_loaders(image_dir, mask_dir, batch_size, image_height, image_width, num_workers=4, pin_memory=True):
    transform = A.Compose([
        A.Resize(image_height, image_width),
        A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    dataset = CTDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader

from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    score = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return score.mean().item()

def iou_score(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score.mean().item()

def hausdorff_distance(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    batch_size = pred.shape[0]
    total_distance = 0.0
    for i in range(batch_size):
        pred_binary = (pred[i, 0] > 0.5).astype(np.uint8)
        target_binary = (target[i, 0] > 0.5).astype(np.uint8)
        pred_coords = np.argwhere(pred_binary)
        target_coords = np.argwhere(target_binary)

        if len(pred_coords) == 0 or len(target_coords) == 0:
            continue

        hd1 = directed_hausdorff(pred_coords, target_coords)[0]
        hd2 = directed_hausdorff(target_coords, pred_coords)[0]
        total_distance += max(hd1, hd2)

    return total_distance / batch_size







