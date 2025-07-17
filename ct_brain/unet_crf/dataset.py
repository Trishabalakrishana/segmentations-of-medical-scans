import os
import numpy as np
import nibabel as nib
import random
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, window=(-1000, 400)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.window = window  # (min_HU, max_HU)

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii")])
        if len(self.images) == 0:
            raise ValueError(f"No NIfTI files found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)  # Same filename

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        try:
            image_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
        except Exception as e:
            raise ValueError(f"Error loading NIfTI files: {e}")

        image = image_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        # Clip HU values and normalize
        hu_min, hu_max = self.window
        image = np.clip(image, hu_min, hu_max)
        image = (image - hu_min) / (hu_max - hu_min)  # Normalize to [0, 1]

        # Random axial slice
        slice_idx = random.randint(0, image.shape[2] - 1)
        image_slice = image[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        # Expand dims to [H, W, 1] for Albumentations
        image_slice = np.expand_dims(image_slice, axis=-1)
        mask_slice = np.expand_dims(mask_slice, axis=-1)

        if self.transform:
            try:
                augmented = self.transform(image=image_slice, mask=mask_slice)
                image_slice = augmented["image"]
                mask_slice = augmented["mask"]
            except Exception as e:
                raise ValueError(f"Transform failed for {img_filename}: {e}")

        # Ensure image and mask are tensors of shape [1, H, W]
        if image_slice.ndim == 2:
            image_slice = np.expand_dims(image_slice, axis=0)
        elif image_slice.ndim == 3 and image_slice.shape[-1] == 1:
            image_slice = np.transpose(image_slice, (2, 0, 1))  # [H, W, 1] -> [1, H, W]

        if mask_slice.ndim == 2:
            mask_slice = np.expand_dims(mask_slice, axis=0)
        elif mask_slice.ndim == 3 and mask_slice.shape[-1] == 1:
            mask_slice = np.transpose(mask_slice, (2, 0, 1))  # [H, W, 1] -> [1, H, W]

        return image_slice, mask_slice
