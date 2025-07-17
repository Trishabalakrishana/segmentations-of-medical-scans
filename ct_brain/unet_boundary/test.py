import os
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from model import ImprovedUNET  # ✅ FIXED: Use ImprovedUNET instead of UNET
from torchvision.utils import save_image
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import nibabel as nib

# ✅ Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/ct_scans"
MASK_DIR = "data/masks"
CHECKPOINT = "/content/unet_ct/unet_ct/my_checkpoint.pth.tar"
SAVE_FOLDER = "saved_test_images"
CSV_FILE = "test_results.csv"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
BATCH_SIZE = 1

# ✅ Transform
transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
    ToTensorV2(),
])

# ✅ Dataset
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
        self.samples = []
        for fname in self.image_files:
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                continue
            try:
                img_nii = nib.load(img_path)
                mask_nii = nib.load(mask_path)
                img_data = img_nii.get_fdata()
                mask_data = mask_nii.get_fdata()
                for i in range(img_data.shape[2]):
                    self.samples.append((img_path, mask_path, i, fname))
            except Exception as e:
                print(f"⚠️ Error loading {fname}: {e}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, slice_idx, fname = self.samples[idx]
        try:
            img = nib.load(img_path).get_fdata()[:, :, slice_idx]
            mask = nib.load(mask_path).get_fdata()[:, :, slice_idx]
            
            # ✅ FIXED: Handle image preprocessing more robustly
            img = np.stack([img], axis=-1)  # Single channel
            mask = (mask > 0).astype(np.uint8)
            
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"].unsqueeze(0).float()
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).unsqueeze(0).float()
                
            return img, mask, f"{fname}_slice{slice_idx}"
        except Exception as e:
            print(f"⚠️ Error processing {fname} slice {slice_idx}: {e}")
            # Return dummy data to avoid crashes
            dummy_img = torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_mask = torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH)
            return dummy_img, dummy_mask, f"{fname}_slice{slice_idx}_ERROR"

# ✅ Metrics
def compute_metrics(pred, target):
    """Compute Dice, IoU, and Hausdorff distance metrics"""
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    pred_bin = (pred > 0.5).float()
    
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    
    # Dice coefficient
    dice = (2 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)
    
    # IoU (Jaccard index)
    iou = intersection / (union + 1e-8)
    
    # Hausdorff distance
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
            print(f"⚠️ Warning: Could not compute Hausdorff distance: {e}")
            hd = 0.0
    else:
        # If both are empty, distance is 0; if one is empty, distance is infinity
        hd = 0.0 if np.array_equal(pred_np, target_np) else float('inf')
    
    return dice.item(), iou.item(), hd

# ✅ Data loader
def get_test_loader():
    """Create test data loader"""
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    
    all_nii_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.nii', '.nii.gz'))]
    if len(all_nii_files) == 0:
        raise ValueError(f"No .nii files found in {IMG_DIR}")
    
    print(f"📁 Found {len(all_nii_files)} CT scan volumes for testing")
    ds = TestDataset(IMG_DIR, MASK_DIR, transform=transform)
    print(f"📊 Total test samples (slices): {len(ds)}")
    
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ✅ Load model
def load_model():
    """Load the trained model from checkpoint"""
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT}")
    
    # ✅ FIXED: Use ImprovedUNET to match training script
    model = ImprovedUNET(in_channels=1, out_channels=1, dropout=0.1).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"✅ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded from {CHECKPOINT}")
    
    model.eval()
    return model

# ✅ Test & Save
def run_test():
    """Run comprehensive testing and save results"""
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
                    
                    # Skip error samples
                    if "ERROR" in filename[0]:
                        continue
                    
                    pred = torch.sigmoid(model(x))
                    loss = F.binary_cross_entropy(pred, y)
                    
                    dice, iou, hd = compute_metrics(pred, y)
                    
                    total_dice += dice
                    total_iou += iou
                    if not np.isinf(hd) and hd < 1000:  # Filter out extremely large values
                        total_hd += hd
                        valid_hd_count += 1
                    total_loss += loss.item()
                    
                    # Save predictions and ground truth
                    pred_save = (pred > 0.5).float()
                    save_image(pred_save, os.path.join(SAVE_FOLDER, f"pred_{idx:03d}.png"))
                    save_image(y, os.path.join(SAVE_FOLDER, f"gt_{idx:03d}.png"))
                    
                    # ✅ FIXED: Handle filename properly (it's a tuple/list)
                    fname = filename[0] if isinstance(filename, (list, tuple)) else filename
                    results.append([fname, dice, iou, hd, loss.item()])
                    
                except Exception as e:
                    print(f"❌ Error processing batch {idx}: {e}")
                    continue
        
        n = len(results)  # Use actual processed samples
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
        
        # Save detailed results to CSV
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Dice", "IoU", "Hausdorff", "BCE_Loss"])
            writer.writerows(results)
        
        print(f"📄 Metrics saved to: {CSV_FILE}")
        
        # ✅ ADDED: Summary statistics
        print(f"\n📈 Summary Statistics:")
        dice_scores = [r[1] for r in results]
        iou_scores = [r[2] for r in results]
        print(f"   Dice - Min: {min(dice_scores):.4f}, Max: {max(dice_scores):.4f}")
        print(f"   IoU  - Min: {min(iou_scores):.4f}, Max: {max(iou_scores):.4f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise

if __name__ == "__main__":
    run_test()