#boundary_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred_logits, target):
        """
        pred_logits: raw output from model (B, 1, H, W)
        target: ground truth mask (B, 1, H, W)
        """
        pred_probs = torch.sigmoid(pred_logits)  # (B, 1, H, W)
        target = target.float()

        loss = 0.0
        for i in range(pred_probs.shape[0]):
            pred_mask = pred_probs[i, 0]
            true_mask = target[i, 0]

            # Compute distance maps
            target_dist = self._distance_transform(true_mask.cpu().numpy())
            pred_dist = self._distance_transform(pred_mask.detach().cpu().numpy() > 0.5)

            target_dist = torch.from_numpy(target_dist).float().to(pred_mask.device)
            pred_dist = torch.from_numpy(pred_dist).float().to(pred_mask.device)

            # L1 distance between boundary maps
            loss += F.l1_loss(pred_mask * target_dist, true_mask * pred_dist)

        return loss / pred_probs.shape[0]

    def _distance_transform(self, mask):
        """
        Compute the distance transform of a binary mask.
        """
        mask = mask.astype(np.uint8)
        if mask.max() == 0:
            return np.zeros_like(mask, dtype=np.float32)
        dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        return dist_out + dist_in
