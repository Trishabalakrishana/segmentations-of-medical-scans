import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        
        if len(self.images) == 0:
            raise ValueError(f"No PNG images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # Replace .png with _mask.png for mask filename
        mask_filename = img_filename.replace(".png", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Check if files exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise ValueError(f"Could not read image {img_path}: {e}")
        
        try:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not read mask {mask_path}: {e}")

        # Validate image and mask shapes
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image, got shape {image.shape} for {img_path}")
        if len(mask.shape) != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape} for {mask_path}")

        # Normalize mask to [0, 1]
        mask /= 255.0

        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            except Exception as e:
                raise ValueError(f"Transform failed for {img_filename}: {e}")

        # Ensure mask has correct dimensions after transform
        if hasattr(mask, 'ndim'):
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)  # Shape becomes (1, H, W)
        else:
            # Handle tensor case (if transform converts to tensor)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # For PyTorch tensors

        return image, mask