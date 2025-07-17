#test.py

import os
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from model import UNET
from torchvision.utils import save_image
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

# ✅ Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
CHECKPOINT = "/content/unet_week3/unet_crf/my_checkpoint.pth.tar"
SAVE_FOLDER = "saved_test_images"
CSV_FILE = "test_results.csv"
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
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("images", "masks").replace(".png", "_mask.png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].float() / 255.0  # ✅ Normalized mask
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return img, mask, os.path.basename(img_path)

# ✅ Metrics
def compute_metrics(pred, target):
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()

    dice = (2 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    pred_np = pred_bin.squeeze().numpy().astype(np.uint8)
    target_np = target.squeeze().numpy().astype(np.uint8)

    pred_pts = np.argwhere(pred_np > 0)
    target_pts = np.argwhere(target_np > 0)

    if pred_pts.size > 0 and target_pts.size > 0:
        try:
            hd1 = directed_hausdorff(pred_pts, target_pts)[0]
            hd2 = directed_hausdorff(target_pts, pred_pts)[0]
            hd = max(hd1, hd2)
        except Exception as e:
            print(f"Warning: Could not compute Hausdorff distance: {e}")
            hd = 0.0
    else:
        hd = 0.0 if np.array_equal(pred_np, target_np) else float('inf')

    return dice.item(), iou.item(), hd

# ✅ Data loader
def get_test_loader():
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_images) == 0:
        raise ValueError(f"No image files found in {IMG_DIR}")

    images = sorted([os.path.join(IMG_DIR, f) for f in all_images])
    test_imgs = images[-min(20, len(images)):]
    print(f"Found {len(test_imgs)} test images")

    ds = TestDataset(test_imgs, transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ✅ Load model
def load_model():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT}")

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"✅ Model loaded from {CHECKPOINT}")
    model.eval()
    return model

# ✅ Test & Save
def run_test():
    try:
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        print(f"📁 Saving results to: {SAVE_FOLDER}")

        model = load_model()
        loader = get_test_loader()

        total_dice, total_iou, total_hd, total_loss = 0, 0, 0, 0
        valid_hd_count = 0
        results = []

        print("🔄 Starting evaluation...")
        with torch.no_grad():
            for idx, (x, y, filename) in enumerate(tqdm(loader, desc="Testing")):
                try:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = torch.sigmoid(model(x))
                    loss = F.binary_cross_entropy(pred, y)

                    dice, iou, hd = compute_metrics(pred, y)
                    total_dice += dice
                    total_iou += iou
                    if not np.isinf(hd):
                        total_hd += hd
                        valid_hd_count += 1
                    total_loss += loss.item()

                    pred_save = (pred > 0.5).float()
                    save_image(pred_save, os.path.join(SAVE_FOLDER, f"pred_{idx:03d}.png"))
                    save_image(y, os.path.join(SAVE_FOLDER, f"gt_{idx:03d}.png"))

                    results.append([filename, dice, iou, hd, loss.item()])
                except Exception as e:
                    print(f"Error processing batch {idx}: {e}")
                    continue

        n = len(loader)
        if n == 0:
            print("❌ No valid samples processed")
            return

        avg_hd = total_hd / valid_hd_count if valid_hd_count > 0 else 0.0

        print(f"\n📊 Final Evaluation on Test Set ({n} samples)")
        print(f"✅ Avg Dice Score: {total_dice/n:.4f}")
        print(f"✅ Avg IoU Score: {total_iou/n:.4f}")
        print(f"✅ Avg Hausdorff Distance: {avg_hd:.4f} (valid: {valid_hd_count}/{n})")
        print(f"✅ Avg BCE Loss: {total_loss/n:.4f}")
        print(f"💾 Results saved in: {SAVE_FOLDER}")

        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Dice", "IoU", "Hausdorff", "BCE_Loss"])
            writer.writerows(results)

        print(f"📄 Metrics saved to: {CSV_FILE}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise

if __name__ == "__main__":
    run_test()
