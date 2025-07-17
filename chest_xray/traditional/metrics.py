import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score
import json
import os
import cv2

class SegmentationMetrics:
    """Metrics computation for classical segmentation methods"""

    def __init__(self):
        self.metrics = {
            "graphcut": {"dice": [], "iou": [], "precision": [], "recall": [], "hausdorff": []},
            "grabcut": {"dice": [], "iou": [], "precision": [], "recall": [], "hausdorff": []},
            "chanvese": {"dice": [], "iou": [], "precision": [], "recall": [], "hausdorff": []},
            "morphacwe": {"dice": [], "iou": [], "precision": [], "recall": [], "hausdorff": []}
        }

    def dice_score(self, true_mask, pred_mask):
        true_mask = (true_mask > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()

        intersection = np.sum(true_flat * pred_flat)
        total = np.sum(true_flat) + np.sum(pred_flat)

        if total == 0:
            return 1.0 if np.sum(pred_flat) == 0 else 0.0

        return 2.0 * intersection / total

    def iou_score(self, true_mask, pred_mask):
        true_mask = (true_mask > 0).astype(np.uint8)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()

        intersection = np.sum(true_flat * pred_flat)
        union = np.sum(true_flat) + np.sum(pred_flat) - intersection

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    def hausdorff_distance(self, true_mask, pred_mask):
        try:
            true_mask = (true_mask > 0).astype(np.uint8)
            pred_mask = (pred_mask > 0).astype(np.uint8)

            true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not true_contours or not pred_contours:
                return np.inf

            true_points = np.vstack([c.reshape(-1, 2) for c in true_contours])
            pred_points = np.vstack([c.reshape(-1, 2) for c in pred_contours])

            hd1 = directed_hausdorff(true_points, pred_points)[0]
            hd2 = directed_hausdorff(pred_points, true_points)[0]

            return max(hd1, hd2)

        except Exception as e:
            print(f"[ERROR] Hausdorff distance failed: {e}")
            return np.inf

    def precision_recall(self, true_mask, pred_mask):
        true_flat = (true_mask > 0).astype(np.uint8).flatten()
        pred_flat = (pred_mask > 0).astype(np.uint8).flatten()

        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)

        return precision, recall

    def add_scores(self, algorithm, true_mask, pred_mask):
        dice = self.dice_score(true_mask, pred_mask)
        iou = self.iou_score(true_mask, pred_mask)
        hausdorff = self.hausdorff_distance(true_mask, pred_mask)
        precision, recall = self.precision_recall(true_mask, pred_mask)

        self.metrics[algorithm]["dice"].append(dice)
        self.metrics[algorithm]["iou"].append(iou)
        self.metrics[algorithm]["hausdorff"].append(hausdorff)
        self.metrics[algorithm]["precision"].append(precision)
        self.metrics[algorithm]["recall"].append(recall)

    def compute_statistics(self):
        print("\n" + "=" * 80)
        print("SEGMENTATION RESULTS - COMPREHENSIVE METRICS")
        print("=" * 80)

        results = {}
        for algo, scores in self.metrics.items():
            if scores["dice"]:
                dice_scores = np.array(scores["dice"])
                iou_scores = np.array(scores["iou"])
                hd_scores = np.array([h for h in scores["hausdorff"] if h != np.inf])
                precision_scores = np.array(scores["precision"])
                recall_scores = np.array(scores["recall"])

                results[algo] = {
                    "dice_mean": np.mean(dice_scores),
                    "dice_std": np.std(dice_scores),
                    "iou_mean": np.mean(iou_scores),
                    "iou_std": np.std(iou_scores),
                    "hausdorff_mean": np.mean(hd_scores) if len(hd_scores) > 0 else np.inf,
                    "hausdorff_std": np.std(hd_scores) if len(hd_scores) > 0 else 0,
                    "precision_mean": np.mean(precision_scores),
                    "precision_std": np.std(precision_scores),
                    "recall_mean": np.mean(recall_scores),
                    "recall_std": np.std(recall_scores),
                    "n_samples": len(dice_scores)
                }

                print(f"\n{algo.upper()}:")
                print(f"  Dice:      {results[algo]['dice_mean']:.4f} ± {results[algo]['dice_std']:.4f}")
                print(f"  IoU:       {results[algo]['iou_mean']:.4f} ± {results[algo]['iou_std']:.4f}")
                print(f"  Hausdorff: {results[algo]['hausdorff_mean']:.2f} ± {results[algo]['hausdorff_std']:.2f}")
                print(f"  Precision: {results[algo]['precision_mean']:.4f} ± {results[algo]['precision_std']:.4f}")
                print(f"  Recall:    {results[algo]['recall_mean']:.4f} ± {results[algo]['recall_std']:.4f}")
                print(f"  Samples:   {results[algo]['n_samples']}")

        with open("outputs/metrics_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print("=" * 80)
        return results


# Global instance for use in pipeline
metrics = SegmentationMetrics()
