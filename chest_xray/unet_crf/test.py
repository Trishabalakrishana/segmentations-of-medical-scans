import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from utils import CombinedLoss, get_loaders
from predict_and_crf import EnhancedSegmentationPredictor, PredictionConfig, apply_crf_to_prediction

# ✅ Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "/content/UNET_CRF/UNET_CRF/data/images"
MASK_DIR = "/content/UNET_CRF/UNET_CRF/data/masks"
CHECKPOINT = "/content/UNET_CRF/UNET_CRF/model_checkpoint.pth.tar"
SAVE_FOLDER = "saved_test_images_crf"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
BATCH_SIZE = 1

# ✅ Transform
transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ToTensorV2(),
])

# ✅ Dataset
class TestDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float() / 255.0
        return img, mask

# ✅ Metrics
def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()

    dice = (2 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8)
    target_np = target.squeeze().cpu().numpy().astype(np.uint8)

    pred_pts = np.argwhere(pred_np > 0)
    target_pts = np.argwhere(target_np > 0)

    if pred_pts.size and target_pts.size:
        hd1 = directed_hausdorff(pred_pts, target_pts)[0]
        hd2 = directed_hausdorff(target_pts, pred_pts)[0]
        hd = max(hd1, hd2)
    else:
        hd = 0.0 if np.array_equal(pred_np, target_np) else float('inf')

    return dice.item(), iou.item(), hd

# ✅ Run test using EnhancedSegmentationPredictor
def run_test():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # 🔧 Define configuration
    config = PredictionConfig(
        model_path=CHECKPOINT,
        model_type="UNET",
        model_channels=(3, 1),
        image_path=IMG_DIR,
        save_dir=SAVE_FOLDER,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        maintain_aspect_ratio=False,

        # Postprocessing + CRF
        use_crf=True,
        crf_detail_level="high",
        confidence_threshold=0.5,
        apply_morphology=True,
        morphology_kernel_size=3,
        morphology_iterations=1,

        # Evaluation
        ground_truth_dir=MASK_DIR,
        calculate_metrics=True,

        # Output
        save_raw_mask=True,
        save_overlay=True,
        save_confidence=False,
        save_probabilities=False,
        overlay_alpha=0.5,
        overlay_color=(0, 255, 0),

        # Misc
        batch_processing=True,
        recursive_search=False,
        device=DEVICE,
        num_workers=2,
        batch_size=BATCH_SIZE
    )

    # 🔍 Load and run
    predictor = EnhancedSegmentationPredictor(config)
    predictor.load_model()
    results = predictor.predict_batch(config.image_path)

    # ✅ Evaluate
    loss_fn = CombinedLoss()
    total_dice, total_iou, total_hd = 0, 0, 0
    total_loss = 0
    total_ce, total_dice_loss, total_boundary = 0, 0, 0

    for name, output in results.items():
        if 'refined_mask' not in output or output['refined_mask'] is None:
            continue

        pred = torch.from_numpy(output['refined_mask'] / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        gt_path = os.path.join(MASK_DIR, name)
        if not os.path.exists(gt_path):
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(gt / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        dice, iou, hd = compute_metrics(pred, gt_tensor)
        total_dice += dice
        total_iou += iou
        total_hd += hd

        loss, losses = loss_fn(pred, gt_tensor)
        total_loss += loss.item()
        total_ce += losses['ce_loss']
        total_dice_loss += losses['dice_loss']
        total_boundary += losses['boundary_loss']

    n = len(results)
    print(f"\n📊 Final Evaluation with CRF on Test Set")
    print(f"✅ Avg Dice Score: {total_dice/n:.4f}")
    print(f"✅ Avg IoU Score: {total_iou/n:.4f}")
    print(f"✅ Avg Hausdorff Distance: {total_hd/n:.4f}")
    print(f"✅ Avg Loss: {total_loss/n:.4f}")
    print(f"  - CE Loss: {total_ce/n:.4f}")
    print(f"  - Dice Loss: {total_dice_loss/n:.4f}")
    print(f"  - Boundary Loss: {total_boundary/n:.4f}")

if __name__ == "__main__":
    run_test()
