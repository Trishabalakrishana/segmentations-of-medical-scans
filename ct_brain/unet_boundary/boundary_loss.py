import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    """
    Improved Boundary Loss: Penalizes discrepancies near the object boundary
    using distance maps for better edge alignment.
    """
    def __init__(self, alpha=1.0):
        super(BoundaryLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = torch.sigmoid(pred_logits)
        target = target.float()
        
        batch_size = pred_probs.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_mask = pred_probs[i, 0]
            true_mask = target[i, 0]
            
            # Convert to numpy for distance transform
            true_binary = (true_mask > 0.5).cpu().numpy().astype(np.uint8)
            
            # Skip if no foreground pixels
            if true_binary.sum() == 0:
                continue
                
            # Compute distance transform only for target boundaries
            true_dist = self._compute_distance_transform(true_binary)
            true_dist = torch.from_numpy(true_dist).float().to(pred_mask.device)
            
            # Boundary loss: penalize predictions far from true boundaries
            boundary_loss = torch.mean(true_dist * torch.abs(pred_mask - true_mask))
            total_loss += boundary_loss
            
        return total_loss / max(batch_size, 1)
    
    def _compute_distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Compute distance transform from object boundaries.
        """
        if binary_mask.max() == 0:
            return np.zeros_like(binary_mask, dtype=np.float32)
        
        # Find boundaries using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        boundaries = binary_mask - eroded
        
        # If no boundaries found, use the mask itself
        if boundaries.sum() == 0:
            boundaries = binary_mask
        
        # Distance transform from boundaries
        dist_transform = distance_transform_edt(1 - boundaries)
        
        # Normalize to prevent very large values
        if dist_transform.max() > 0:
            dist_transform = dist_transform / (dist_transform.max() + 1e-8)
        
        return dist_transform.astype(np.float32)


class DiceLoss(nn.Module):
    """
    Improved Dice Loss with better numerical stability.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = torch.sigmoid(pred_logits)
        target = target.float()
        
        # Flatten tensors
        pred_flat = pred_probs.view(pred_probs.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance in segmentation.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Improved Combined Loss with better weighting and additional components.
    """
    def __init__(self, ce_weight=1.0, dice_weight=2.0, boundary_weight=1.0, 
                 focal_weight=0.5, use_focal=True):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        self.use_focal = use_focal
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        if use_focal:
            self.focal = FocalLoss()
    
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.shape != targets.shape:
            targets = targets.float()
        
        losses = {}
        
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        losses['bce'] = bce_loss
        
        # Dice Loss
        dice_loss = self.dice(inputs, targets)
        losses['dice'] = dice_loss
        
        # Boundary Loss
        boundary_loss = self.boundary(inputs, targets)
        losses['boundary'] = boundary_loss
        
        # Focal Loss (optional)
        if self.use_focal:
            focal_loss = self.focal(inputs, targets)
            losses['focal'] = focal_loss
        
        # Combined loss
        total = (self.ce_weight * bce_loss +
                 self.dice_weight * dice_loss +
                 self.boundary_weight * boundary_loss)
        
        if self.use_focal:
            total += self.focal_weight * focal_loss
        
        return total, losses


class SegmentationMetrics:
    """
    Helper class to compute segmentation metrics.
    """
    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).float()
        target = target.float()
        
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).float()
        target = target.float()
        
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor):
        """
        Compute Hausdorff distance between predicted and target masks.
        """
        try:
            pred = torch.sigmoid(pred)
            pred_binary = (pred > 0.5).cpu().numpy().astype(np.uint8)
            target_binary = target.cpu().numpy().astype(np.uint8)
            
            # If either mask is empty, return large distance
            if pred_binary.sum() == 0 or target_binary.sum() == 0:
                return 100.0
            
            # Find contours
            pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(pred_contours) == 0 or len(target_contours) == 0:
                return 100.0
            
            # Get contour points
            pred_points = np.vstack(pred_contours).squeeze()
            target_points = np.vstack(target_contours).squeeze()
            
            if pred_points.ndim == 1:
                pred_points = pred_points.reshape(1, -1)
            if target_points.ndim == 1:
                target_points = target_points.reshape(1, -1)
            
            # Compute Hausdorff distance
            dist1 = np.max(np.min(np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=2), axis=1))
            dist2 = np.max(np.min(np.linalg.norm(target_points[:, None] - pred_points[None, :], axis=2), axis=1))
            
            return max(dist1, dist2)
        except:
            return 100.0


# Example usage with improved loss weights
def get_loss_function():
    """
    Factory function to create the loss function with optimized weights.
    """
    return CombinedLoss(
        ce_weight=0.5,      # Reduced BCE weight
        dice_weight=2.0,    # Increased Dice weight for better overlap
        boundary_weight=1.5, # Increased boundary weight for better edges
        focal_weight=0.3,   # Focal loss for hard examples
        use_focal=True
    )