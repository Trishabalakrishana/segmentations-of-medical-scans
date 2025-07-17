#utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

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

# ✅ Add data loader utility
from dataset import ChestXrayDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

def get_loaders(image_dir, mask_dir, batch_size, image_height, image_width, num_workers=4, pin_memory=True):
    transform = A.Compose([
        A.Resize(image_height, image_width),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    dataset = ChestXrayDataset(
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
